"""Fast four-component validation on the legacy synthetic BTT data.

This script uses the current project's `BTTSequenceDataset` to generate the
existing four constant-frequency components, then trains a candidate-basin
encoder with one candidate posterior per component.

It is a checkpoint-transfer validation harness, not the final full mixture
ELBO. The training objective is intentionally practical for CPU:

- supervised/soft-target basin loss per component;
- supervised local frequency posterior loss inside the label basin;
- joint constant-frequency reconstruction loss using all four refined
  frequencies and a closed-form complex LS amplitude solve.

The decoder is joint over the four components, so it exercises the part that
the single-component smoke test cannot cover.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LEGACY_DIR = Path(__file__).resolve().parent / "legacy_reuse" / "current_project"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

from config import CONFIG  # noqa: E402
from dataset import BTTSequenceDataset  # noqa: E402
from synthesis_dataset import compute_frequency_support  # noqa: E402
from bse_btt.inference.likelihood import bse_log_likelihood  # noqa: E402
from bse_btt.wavelet.nonuniform_dwt import nonuniform_gabor_spectrogram  # noqa: E402


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    return (values - np.mean(values)) / (np.std(values) + eps)


def masked_mean(h, mask):
    w = mask.float().unsqueeze(-1)
    return (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def profile_scores_single_tone(y_complex, t, grid_hz, ridge=1e-6):
    phase = 2.0 * np.pi * grid_hz[:, None] * t[None, :]
    phi = np.exp(1j * phase)
    denom = np.sum(np.conj(phi) * phi, axis=1).real + ridge
    amp = np.sum(np.conj(phi) * y_complex[None, :], axis=1) / np.maximum(denom, 1e-12)
    residual = y_complex[None, :] - amp[:, None] * phi
    sse = np.sum(np.abs(residual) ** 2, axis=1)
    return -sse


def subtract_single_tone(y_complex, t, freq_hz, ridge=1e-6):
    phase = 2.0 * np.pi * float(freq_hz) * t
    phi = np.exp(1j * phase)
    denom = np.sum(np.conj(phi) * phi).real + ridge
    amp = np.sum(np.conj(phi) * y_complex) / max(denom, 1e-12)
    return y_complex - amp * phi


def residual_after_joint_tones(y_complex, t, freqs_hz, ridge=1e-6):
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    if freqs_hz.size == 0:
        return y_complex
    phase = 2.0 * np.pi * t[:, None] * freqs_hz[None, :]
    phi = np.exp(1j * phase)
    gram = phi.conj().T @ phi
    rhs = phi.conj().T @ y_complex
    amp = np.linalg.solve(gram + ridge * np.eye(freqs_hz.size), rhs)
    return y_complex - phi @ amp


def joint_profile_sse(y_complex, t, freqs_hz, ridge=1e-6):
    residual = residual_after_joint_tones(y_complex, t, freqs_hz, ridge=ridge)
    return float(np.sum(np.abs(residual) ** 2))


def basis_correlation(f_i, f_j, t):
    phase = 2.0 * np.pi * (float(f_j) - float(f_i)) * t
    return float(np.abs(np.mean(np.exp(1j * phase))))


def alias_family_match(freq, rep_freq, t, args, y_complex=None):
    corr = basis_correlation(freq, rep_freq, t)
    if corr < float(args.basin_alias_corr_threshold):
        return False
    if y_complex is None:
        return True
    r1 = joint_profile_sse(
        y_complex,
        t,
        [rep_freq],
        ridge=args.profile_ridge,
    )
    r2 = joint_profile_sse(
        y_complex,
        t,
        [rep_freq, freq],
        ridge=args.profile_ridge,
    )
    rel_drop = (r1 - r2) / max(r1, 1e-12)
    return rel_drop <= float(args.basin_alias_max_residual_drop_ratio)


def local_peak_indices(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    if scores.size < 3:
        return np.arange(scores.size, dtype=np.int64)
    peaks = np.where((scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:]))[0] + 1
    if scores[0] >= scores[1]:
        peaks = np.r_[0, peaks]
    if scores[-1] >= scores[-2]:
        peaks = np.r_[peaks, scores.size - 1]
    return peaks.astype(np.int64)


def group_alias_families(freqs, scores, t, args, y_complex=None):
    freqs = np.asarray(freqs, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if freqs.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    families = []
    for idx in order:
        freq = float(freqs[idx])
        score = float(scores[idx])
        placed = False
        for family in families:
            if alias_family_match(
                freq,
                family["rep_freq"],
                t,
                args,
                y_complex=y_complex,
            ):
                family["members"].append(freq)
                family["member_scores"].append(score)
                if score > family["score"]:
                    family["rep_freq"] = freq
                    family["score"] = score
                placed = True
                break
        if not placed:
            families.append(
                {
                    "rep_freq": freq,
                    "score": score,
                    "members": [freq],
                    "member_scores": [score],
                }
            )
    spacing = (
        float(args.global_fmax_hz - args.global_fmin_hz) / max(int(args.global_bins) - 1, 1)
    )
    min_half_width = max(
        float(args.local_radius_hz),
        float(args.basin_min_half_width_hz),
        float(args.basin_half_width_bins) * spacing,
    )
    for family in families:
        family["radius"] = min_half_width
    families.sort(key=lambda row: row["score"], reverse=True)
    return families


def profile_basin_families(y_complex, t, args, max_peaks=None):
    grid = np.linspace(
        float(args.global_fmin_hz),
        float(args.global_fmax_hz),
        int(args.global_bins),
    )
    scores = profile_scores_single_tone(y_complex, t, grid, ridge=args.profile_ridge)
    peaks = local_peak_indices(scores)
    if peaks.size == 0:
        peaks = np.arange(grid.size, dtype=np.int64)
    peak_order = peaks[np.argsort(scores[peaks])[::-1]]
    if max_peaks is not None:
        peak_order = peak_order[: int(max_peaks)]
    return group_alias_families(
        grid[peak_order],
        scores[peak_order],
        t,
        args,
        y_complex=y_complex,
    )


def score_families_by_joint_residual(y_complex, t, selected_freqs, families, args):
    current_sse = joint_profile_sse(
        y_complex,
        t,
        selected_freqs,
        ridge=args.profile_ridge,
    )
    rows = []
    for row in families:
        member_rows = []
        member_scores = row.get("member_scores", [row["score"]] * len(row["members"]))
        for member_freq, member_profile_score in zip(row["members"], member_scores):
            freq = float(member_freq)
            if any(
                alias_family_match(
                    freq,
                    old,
                    t,
                    args,
                    y_complex=y_complex,
                )
                for old in selected_freqs
            ):
                continue
            next_freqs = list(selected_freqs) + [freq]
            next_sse = joint_profile_sse(
                y_complex,
                t,
                next_freqs,
                ridge=args.profile_ridge,
            )
            member_rows.append(
                {
                    "rep_freq": freq,
                    "score": current_sse - next_sse,
                    "single_profile_score": float(member_profile_score),
                    "joint_residual_sse": next_sse,
                    "joint_improvement": current_sse - next_sse,
                    "members": [freq],
                    "member_scores": [float(member_profile_score)],
                    "radius": float(row["radius"]),
                    "selected": False,
                }
            )
        if not member_rows:
            continue
        member_rows.sort(key=lambda item: item["score"], reverse=True)
        scored = dict(member_rows[0])
        scored["members"] = list(row["members"])
        scored["member_scores"] = list(member_scores)
        scored["member_rows"] = member_rows
        rows.append(scored)
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def refine_joint_frequencies(y_complex, t, freqs_hz, args):
    freqs = [float(freq) for freq in freqs_hz]
    if not freqs:
        return freqs
    spacing = (
        float(args.global_fmax_hz - args.global_fmin_hz) / max(int(args.global_bins) - 1, 1)
    )
    half_width = max(float(args.local_radius_hz), float(args.basin_half_width_bins) * spacing)
    n_grid = max(int(args.basin_refine_bins), 1)
    for _ in range(max(int(args.basin_refine_passes), 0)):
        for idx in range(len(freqs)):
            if n_grid == 1:
                grid = np.asarray([freqs[idx]], dtype=np.float64)
            else:
                lo = max(float(args.global_fmin_hz), freqs[idx] - half_width)
                hi = min(float(args.global_fmax_hz), freqs[idx] + half_width)
                grid = np.linspace(lo, hi, n_grid)
            best_freq = freqs[idx]
            best_sse = float("inf")
            for candidate in grid:
                trial = list(freqs)
                trial[idx] = float(candidate)
                sse = joint_profile_sse(y_complex, t, trial, ridge=args.profile_ridge)
                if sse < best_sse:
                    best_sse = sse
                    best_freq = float(candidate)
            freqs[idx] = best_freq
    return freqs


def dedupe_basin_rows(rows, t, args):
    ordered = sorted(
        rows,
        key=lambda row: (1 if row.get("selected", False) else 0, float(row["score"])),
        reverse=True,
    )
    kept = []
    for row in ordered:
        freq = float(row["rep_freq"])
        if any(
            alias_family_match(
                freq,
                old["rep_freq"],
                t,
                args,
            )
            for old in kept
        ):
            continue
        kept.append(row)
    return kept


def dedupe_frequency_rows(rows, args):
    spacing = (
        float(args.global_fmax_hz - args.global_fmin_hz) / max(int(args.global_bins) - 1, 1)
    )
    min_gap = max(float(args.basin_candidate_min_gap_bins) * spacing, 1e-9)
    ordered = sorted(
        rows,
        key=lambda row: (1 if row.get("selected", False) else 0, float(row["score"])),
        reverse=True,
    )
    kept = []
    for row in ordered:
        freq = float(row["rep_freq"])
        if any(abs(freq - float(old["rep_freq"])) <= min_gap for old in kept):
            continue
        kept.append(row)
    return kept


def make_basin_row(freq, score, radius):
    return {
        "rep_freq": float(freq),
        "score": float(score),
        "single_profile_score": float(score),
        "joint_residual_sse": np.nan,
        "joint_improvement": np.nan,
        "members": [float(freq)],
        "member_scores": [float(score)],
        "radius": float(radius),
        "selected": False,
    }


def fill_basin_rows_to_top_j(rows, y_complex, t, args):
    if len(rows) >= int(args.top_j):
        return rows
    grid = np.linspace(
        float(args.global_fmin_hz),
        float(args.global_fmax_hz),
        int(args.global_bins),
    )
    scores = profile_scores_single_tone(y_complex, t, grid, ridge=args.profile_ridge)
    spacing = (
        float(args.global_fmax_hz - args.global_fmin_hz) / max(int(args.global_bins) - 1, 1)
    )
    radius = max(
        float(args.local_radius_hz),
        float(args.basin_min_half_width_hz),
        float(args.basin_half_width_bins) * spacing,
    )
    padded = list(rows)
    for idx in np.argsort(scores)[::-1]:
        freq = float(grid[idx])
        if any(abs(freq - float(row["rep_freq"])) <= 1e-9 for row in padded):
            continue
        padded.append(make_basin_row(freq, scores[idx], radius))
        if len(padded) >= int(args.top_j):
            break
    if len(padded) < int(args.top_j):
        for freq in grid:
            padded.append(make_basin_row(freq, -np.inf, radius))
            if len(padded) >= int(args.top_j):
                break
    return padded


def make_profile_basin_candidates(y_complex, t, true_freq, args):
    families = profile_basin_families(y_complex, t, args, max_peaks=args.basin_peak_count)
    freqs = [row["rep_freq"] for row in families[: int(args.top_j)]]
    scores = [row["score"] for row in families[: int(args.top_j)]]
    radius = [row["radius"] for row in families[: int(args.top_j)]]
    if len(freqs) < int(args.top_j):
        fallback = profile_basin_families(y_complex, t, args, max_peaks=None)
        for row in fallback:
            if any(
                alias_family_match(row["rep_freq"], f, t, args)
                for f in freqs
            ):
                continue
            freqs.append(row["rep_freq"])
            scores.append(row["score"])
            radius.append(row["radius"])
            if len(freqs) >= int(args.top_j):
                break
    if not freqs:
        raise RuntimeError("No profile basin candidates were generated")
    freqs = np.asarray(freqs[: int(args.top_j)], dtype=np.float64)
    scores = np.asarray(scores[: int(args.top_j)], dtype=np.float64)
    radius = np.asarray(radius[: int(args.top_j)], dtype=np.float64)
    order = np.argsort(freqs)
    freqs = freqs[order]
    scores = scores[order]
    radius = radius[order]
    label = int(np.argmin(np.abs(freqs - true_freq)))
    best_freq = float(freqs[int(np.argmax(scores))])
    return freqs, radius, scores, label, best_freq


def make_greedy_basin_pool_candidates(y_complex, t, args):
    selected = []
    pool = []
    for round_idx in range(int(args.basin_greedy_rounds)):
        residual = residual_after_joint_tones(
            y_complex,
            t,
            selected,
            ridge=args.profile_ridge,
        )
        families = profile_basin_families(
            residual,
            t,
            args,
            max_peaks=args.basin_peak_count,
        )
        scored = score_families_by_joint_residual(
            y_complex,
            t,
            selected,
            families,
            args,
        )
        if not scored:
            break
        for row in scored[: int(args.basin_round_candidates)]:
            row = dict(row)
            row["round"] = round_idx
            pool.append(row)
            for member in row.get("member_rows", [])[: int(args.basin_alias_members_per_family)]:
                member = dict(member)
                member["round"] = round_idx
                pool.append(member)
        selected.append(scored[0]["rep_freq"])
        selected = refine_joint_frequencies(y_complex, t, selected, args)
        selected_sse = joint_profile_sse(
            y_complex,
            t,
            selected,
            ridge=args.profile_ridge,
        )
        for freq in selected:
            pool.append(
                {
                    "rep_freq": float(freq),
                    "score": -selected_sse,
                    "single_profile_score": np.nan,
                    "joint_residual_sse": selected_sse,
                    "joint_improvement": np.nan,
                    "members": [float(freq)],
                    "radius": max(
                        float(args.local_radius_hz),
                        float(args.basin_min_half_width_hz),
                        float(args.basin_half_width_bins)
                        * float(args.global_fmax_hz - args.global_fmin_hz)
                        / max(int(args.global_bins) - 1, 1),
                    ),
                    "round": round_idx,
                    "selected": True,
                }
            )
    grouped = dedupe_frequency_rows(pool, args)
    if len(grouped) < int(args.top_j):
        fallback = profile_basin_families(y_complex, t, args, max_peaks=None)
        fallback = score_families_by_joint_residual(
            y_complex,
            t,
            selected,
            fallback,
            args,
        )
        expanded_fallback = []
        for row in fallback:
            expanded_fallback.append(row)
            expanded_fallback.extend(
                row.get("member_rows", [])[: int(args.basin_alias_members_per_family)]
            )
        grouped = dedupe_frequency_rows(grouped + expanded_fallback, args)
    grouped = fill_basin_rows_to_top_j(grouped, y_complex, t, args)
    grouped = grouped[: int(args.top_j)]
    freqs = np.asarray([row["rep_freq"] for row in grouped], dtype=np.float64)
    scores = np.asarray([row["score"] for row in grouped], dtype=np.float64)
    radius = np.asarray([row["radius"] for row in grouped], dtype=np.float64)
    order = np.argsort(freqs)
    return freqs[order], radius[order], scores[order]


def make_component_candidates(y_complex, t, true_freq, lower, upper, args):
    grid = np.linspace(float(lower), float(upper), int(args.grid_size))
    scores = profile_scores_single_tone(y_complex, t, grid, ridge=args.profile_ridge)
    top_j = int(args.top_j)
    keep = np.argsort(scores)[-top_j:][::-1]
    nearest_grid = int(np.argmin(np.abs(grid - true_freq)))
    if nearest_grid not in keep:
        keep[-1] = nearest_grid
    keep = keep[np.argsort(grid[keep])]
    freqs = grid[keep]
    kept_scores = scores[keep]
    label = int(np.argmin(np.abs(freqs - true_freq)))
    diffs = np.diff(freqs)
    if diffs.size:
        left = np.r_[diffs[0], diffs]
        right = np.r_[diffs, diffs[-1]]
        radius = 0.45 * np.minimum(left, right)
    else:
        radius = np.full_like(freqs, args.local_radius_hz)
    radius = np.maximum(radius, float(args.local_radius_hz))
    best_freq = float(freqs[int(np.argmax(kept_scores))])
    return freqs, radius, kept_scores, label, best_freq


def make_global_fmax_candidates(y_complex, t, true_freq, args):
    grid = np.linspace(
        float(args.global_fmin_hz),
        float(args.global_fmax_hz),
        int(args.global_bins),
    )
    scores = profile_scores_single_tone(y_complex, t, grid, ridge=args.profile_ridge)
    if grid.size > 1:
        spacing = float(np.median(np.diff(grid)))
    else:
        spacing = float(args.local_radius_hz)
    radius = np.full_like(grid, max(0.5 * spacing, float(args.local_radius_hz)))
    label = int(np.argmin(np.abs(grid - true_freq)))
    best_freq = float(grid[int(np.argmax(scores))])
    return grid, radius, scores, label, best_freq


def alias_scores_constant_frequency(freqs, y_complex, t, sample_rate_hz, args):
    """Score candidates by consistency across simple sampling aliases.

    This is a constant-frequency adaptation of the bundle's alias-family idea:
    for each physical candidate f, evaluate nearby observed aliases
    |m * fs +/- f| with a direct nonuniform matched-filter score, then keep the
    strongest family response.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    sample_rate_hz = max(float(sample_rate_hz), 1e-8)
    min_hz = float(args.alias_obs_min_hz)
    max_hz = float(args.alias_obs_max_hz)
    out = []
    for freq in freqs:
        family = [float(abs(freq))]
        for order in range(1, int(args.alias_order_max) + 1):
            family.append(abs(order * sample_rate_hz - freq))
            family.append(abs(order * sample_rate_hz + freq))
        best = -np.inf
        for obs_freq in family:
            if obs_freq < min_hz or obs_freq > max_hz:
                continue
            phase = 2.0 * np.pi * obs_freq * t
            basis = np.exp(-1j * phase)
            score = np.abs(np.sum(y_complex * basis)) ** 2 / max(len(t), 1)
            if score > best:
                best = float(score)
        out.append(best if np.isfinite(best) else 0.0)
    return np.asarray(out, dtype=np.float64)


def bse_scores_constant_frequency(freqs, y_complex, t, x_features, probe_angles_rad, args):
    """BSE alias-template likelihood collapsed over time for constant-frequency bins."""
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    y_real = np.real(y_complex).astype(np.float64)
    u_grid = np.linspace(float(t.min()), float(t.max()), int(args.bse_time_bins))
    axis_hz = np.linspace(
        float(args.global_fmin_hz),
        float(args.global_fmax_hz),
        int(args.bse_axis_bins),
    )
    omega_axis = 2.0 * np.pi * axis_hz
    candidate_omega = 2.0 * np.pi * freqs
    speed_norm = np.asarray(x_features[:, 5], dtype=np.float64)
    rot_hz = speed_norm * float(CONFIG["data"]["base_freq"])
    rot_omega_u = 2.0 * np.pi * np.interp(u_grid, t, rot_hz)
    spec = nonuniform_gabor_spectrogram(
        t,
        y_real,
        u_grid,
        omega_axis,
        window_s=max(float(t.max() - t.min()) / max(float(args.bse_window_divisor), 1.0), 1e-6),
        normalize=True,
    )
    ell, _ = bse_log_likelihood(
        spec,
        omega_axis,
        candidate_omega,
        rot_omega_u,
        probe_angles_rad,
        k_min=args.bse_k_min,
        k_max=args.bse_k_max,
        sigma_alias=2.0 * np.pi * args.bse_sigma_hz,
        sigma_scale=args.bse_sigma_scale,
    )
    return np.sum(ell, axis=0)


def make_candidate_features(
    freqs,
    scores,
    alias_scores,
    bse_scores,
    comp_idx,
    lower,
    upper,
    num_components,
):
    freq_norm = 2.0 * (freqs - lower) / max(upper - lower, 1e-8) - 1.0
    score_norm = normalized(scores)
    alias_norm = normalized(alias_scores)
    bse_norm = normalized(bse_scores)
    rank = np.argsort(np.argsort(-scores)).astype(np.float64)
    rank_norm = rank / max(len(freqs) - 1, 1)
    comp_norm = np.full_like(freq_norm, comp_idx / max(num_components - 1, 1))
    return np.column_stack(
        [freq_norm, score_norm, alias_norm, bse_norm, rank_norm, comp_norm]
    ).astype(np.float32)


class LegacyFourComponentCandidateDataset(Dataset):
    def __init__(self, split, n_samples, seed, args):
        data_cfg = CONFIG["data"]
        signal_cfg = CONFIG["signal"]
        freq_cfg = CONFIG["frequency"]
        amp_real_center = signal_cfg["amp_real_center_m"]
        amp_imag_center = signal_cfg["amp_imag_center_m"]
        if float(args.controlled_spacing_hz) > 0.0:
            # Controlled-spacing regime: place num_components tones as a symmetric
            # comb of fixed gap controlled_spacing_hz around center_band_hz, for
            # two-tone / close-spacing resolution studies. spacing_jitter_hz adds
            # an optional per-component absolute jitter (kept < spacing/2 so the
            # supports never overlap); 0 means exactly fixed frequencies.
            n_comp = int(args.num_components)
            spacing = float(args.controlled_spacing_hz)
            center = float(args.center_band_hz)
            centers = center + spacing * (np.arange(n_comp) - (n_comp - 1) / 2.0)
            jit = min(float(args.spacing_jitter_hz), 0.45 * spacing)
            freq_lower = np.maximum(centers - jit, 1e-6)
            freq_upper = centers + jit
            freq_center = centers
            amp_real = np.asarray(signal_cfg["amp_real_center_m"], dtype=np.float64)
            amp_imag = np.asarray(signal_cfg["amp_imag_center_m"], dtype=np.float64)
            recycle = np.arange(n_comp) % len(amp_real)
            amp_real_center = amp_real[recycle].tolist()
            amp_imag_center = amp_imag[recycle].tolist()
        else:
            freq_lower, freq_upper, freq_center, _ = compute_frequency_support(
                freq_center_hz=freq_cfg["center_hz"],
                relative_half_band=freq_cfg["relative_half_band"],
            )
        search_half = float(args.search_relative_half_band) * freq_center
        self.search_lower = np.maximum(freq_center - search_half, 1e-6)
        self.search_upper = freq_center + search_half
        self.num_components = len(freq_center)
        if args.candidate_support_mode == "global_fmax":
            self.top_j = int(args.global_bins)
        elif args.candidate_support_mode == "basin":
            self.top_j = int(args.top_j)
        else:
            self.top_j = int(args.top_j)
        self.args = args
        self.base = BTTSequenceDataset(
            num_sequences=n_samples,
            num_cycles=args.num_cycles,
            num_probes=data_cfg["num_probes"],
            base_freq=data_cfg["base_freq"],
            fluctuation_delta=data_cfg["fluctuation_delta"],
            probe_angles=data_cfg["probes"],
            freq_lower=freq_lower,
            freq_upper=freq_upper,
            amp_real_center=amp_real_center,
            amp_imag_center=amp_imag_center,
            amp_relative_half_band=signal_cfg["amp_data_prior"]["relative_half_band"],
            amp_min_half_band=signal_cfg["amp_data_prior"]["min_half_band_m"],
            snr_db=args.snr_db,
            seed=seed,
            normalization=data_cfg.get("normalization", "per_sequence_std"),
            speed_profile=args.speed_profile,
            speed_change=args.speed_change,
        )
        self.split = split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        x = sample["x"].numpy().astype(np.float32)
        t = sample["t"].numpy().astype(np.float64)
        if getattr(self.args, "include_local_time", False):
            # #4 tier-1: give the encoder the (relative) sampling time directly.
            # The base 6-dim features only expose rev_norm + probe angle + speed,
            # so the exact irregular sample spacing that de-aliases is hidden from
            # the encoder. Append TWO channels (input_dim 6->8):
            #   ch7 t_local = (t - t0)/(t_max - t0) in [0,1]  -> absolute position
            #   ch8 dt_norm = (t_n - t_{n-1})/median(dt)      -> local spacing,
            #        which directly exposes the non-uniform sampling / phase
            #        increments (2*pi*f*dt) that carry the de-aliasing structure.
            n = t.shape[0]
            span = float(t[-1] - t[0]) if n > 1 else 0.0
            t_local = (t - t[0]) / (span + 1e-12)
            dt = np.empty_like(t)
            if n > 1:
                dt[1:] = np.diff(t)
                med = float(np.median(dt[1:]))
                med = med if med > 0 else 1.0
                dt[0] = med  # neutral value for the first (undefined) gap
            else:
                med = 1.0
                dt[0] = med
            dt_norm = dt / (med + 1e-12)
            extra = np.stack([t_local, dt_norm], axis=-1).astype(np.float32)
            x = np.concatenate([x, extra], axis=-1)
        target = sample["target"].numpy().astype(np.float32)
        y_complex = target[:, 0].astype(np.float64) + 1j * target[:, 1].astype(np.float64)
        true_freq = sample["true_freq_hz"].numpy().astype(np.float64)
        k_count = self.num_components
        candidate_freq = np.zeros((k_count, self.top_j), dtype=np.float32)
        local_radius = np.zeros((k_count, self.top_j), dtype=np.float32)
        candidate_features = np.zeros((k_count, self.top_j, 6), dtype=np.float32)
        ell = np.zeros((k_count, self.top_j), dtype=np.float32)
        alias_score = np.zeros((k_count, self.top_j), dtype=np.float32)
        bse_score = np.zeros((k_count, self.top_j), dtype=np.float32)
        labels = np.zeros((k_count,), dtype=np.int64)
        residual = y_complex.copy()
        selected_freqs = []
        sample_rate_hz = 1.0 / max(float(np.median(np.diff(t))), 1e-8)
        probe_angles_rad = np.radians(CONFIG["data"]["probes"])

        def store_component(k, freqs, radius, scores, label, score_signal):
            if self.args.candidate_mode in ("alias_hybrid", "bse_hybrid"):
                aliases = alias_scores_constant_frequency(
                    freqs=freqs,
                    y_complex=score_signal,
                    t=t,
                    sample_rate_hz=sample_rate_hz,
                    args=self.args,
                )
            else:
                aliases = np.zeros_like(freqs, dtype=np.float64)
            if self.args.candidate_mode == "bse_hybrid":
                bse_scores = bse_scores_constant_frequency(
                    freqs=freqs,
                    y_complex=score_signal,
                    t=t,
                    x_features=x,
                    probe_angles_rad=probe_angles_rad,
                    args=self.args,
                )
            else:
                bse_scores = np.zeros_like(freqs, dtype=np.float64)
            candidate_freq[k] = freqs.astype(np.float32)
            local_radius[k] = radius.astype(np.float32)
            ell[k] = scores.astype(np.float32)
            alias_score[k] = aliases.astype(np.float32)
            bse_score[k] = bse_scores.astype(np.float32)
            labels[k] = label
            if self.args.candidate_support_mode in ("global_fmax", "basin"):
                feature_lower = float(self.args.global_fmin_hz)
                feature_upper = float(self.args.global_fmax_hz)
            else:
                feature_lower = self.search_lower[k]
                feature_upper = self.search_upper[k]
            candidate_features[k] = make_candidate_features(
                freqs=freqs,
                scores=scores,
                alias_scores=aliases,
                bse_scores=bse_scores,
                comp_idx=k,
                lower=feature_lower,
                upper=feature_upper,
                num_components=k_count,
            )

        if self.args.candidate_support_mode == "basin":
            freqs, radius, scores = make_greedy_basin_pool_candidates(
                y_complex,
                t,
                self.args,
            )
            for k in range(k_count):
                label = int(np.argmin(np.abs(freqs - true_freq[k])))
                store_component(k, freqs, radius, scores, label, y_complex)
        elif self.args.candidate_search_mode == "serial_global":
            remaining = list(range(k_count))
            while remaining:
                best = None
                for k in remaining:
                    if self.args.candidate_support_mode == "global_fmax":
                        pack = make_global_fmax_candidates(
                            y_complex=residual,
                            t=t,
                            true_freq=true_freq[k],
                            args=self.args,
                        )
                    else:
                        pack = make_component_candidates(
                            y_complex=residual,
                            t=t,
                            true_freq=true_freq[k],
                            lower=self.search_lower[k],
                            upper=self.search_upper[k],
                            args=self.args,
                        )
                    freqs, radius, scores, label, best_freq = pack
                    best_score = float(np.max(scores))
                    if best is None or best_score > best["score"]:
                        best = {
                            "k": k,
                            "pack": pack,
                            "score": best_score,
                            "best_freq": best_freq,
                        }
                k = best["k"]
                freqs, radius, scores, label, best_freq = best["pack"]
                store_component(k, freqs, radius, scores, label, residual)
                selected_freqs.append(best_freq)
                residual = residual_after_joint_tones(
                    y_complex,
                    t,
                    selected_freqs,
                    ridge=self.args.profile_ridge,
                )
                remaining.remove(k)
        else:
            for k in range(k_count):
                search_signal = (
                    residual
                    if self.args.candidate_search_mode == "serial"
                    else y_complex
                )
                if self.args.candidate_support_mode == "global_fmax":
                    freqs, radius, scores, label, best_freq = make_global_fmax_candidates(
                        y_complex=search_signal,
                        t=t,
                        true_freq=true_freq[k],
                        args=self.args,
                    )
                else:
                    freqs, radius, scores, label, best_freq = make_component_candidates(
                        y_complex=search_signal,
                        t=t,
                        true_freq=true_freq[k],
                        lower=self.search_lower[k],
                        upper=self.search_upper[k],
                        args=self.args,
                    )
                store_component(k, freqs, radius, scores, label, search_signal)
                if self.args.candidate_search_mode == "serial":
                    selected_freqs.append(best_freq)
                    residual = residual_after_joint_tones(
                        y_complex,
                        t,
                        selected_freqs,
                        ridge=self.args.profile_ridge,
                    )
        return {
            "x": torch.as_tensor(x, dtype=torch.float32),
            "mask": torch.ones(x.shape[0], dtype=torch.bool),
            "t": sample["t"],
            "target": sample["target"],
            "candidate_features": torch.as_tensor(candidate_features, dtype=torch.float32),
            "candidate_freq_hz": torch.as_tensor(candidate_freq, dtype=torch.float32),
            "local_radius_hz": torch.as_tensor(local_radius, dtype=torch.float32),
            "ell": torch.as_tensor(ell, dtype=torch.float32),
            "alias_score": torch.as_tensor(alias_score, dtype=torch.float32),
            "bse_score": torch.as_tensor(bse_score, dtype=torch.float32),
            "label": torch.as_tensor(labels, dtype=torch.long),
            "true_freq_hz": sample["true_freq_hz"],
            "noise_var_norm": sample["noise_var_norm"],
            "probe_ids": sample["probe_ids"],
        }


class RoPETimeEncoderLayer(nn.Module):
    """Transformer encoder layer with rotary position encoding keyed by real time.

    Standard RoPE rotates Q/K by an angle proportional to the (integer) token
    index, making attention a function of the index gap. Here the angle is
    proportional to the real (local) sampling time t_n, so the post-rotation
    score q_m . k_n depends on the *relative sampling time* (t_m - t_n) -- exactly
    the quantity the tone phase difference 2*pi*f*(t_m - t_n) lives in, and
    translation-invariant in the arbitrary clock origin. Masking is effectively a
    no-op in this project (patches are full length), but it is handled for safety.
    """

    def __init__(self, hidden_dim, nhead, dim_feedforward, dropout, omega):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by nhead for RoPE")
        self.nhead = int(nhead)
        self.head_dim = hidden_dim // nhead
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE pair rotation")
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # omega: [head_dim // 2] angular frequencies (rad / s)
        self.register_buffer("omega", omega)

    def _rotate(self, x, t_local):
        # x: [B, nhead, L, head_dim]; t_local: [B, L] seconds (patch-relative)
        ang = t_local[:, None, :, None] * self.omega  # [B, 1, L, head_dim // 2]
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rot.flatten(-2)

    def forward(self, x, t_local, key_padding_mask=None):
        b, l, h = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, nhead, L, head_dim]
        q = self._rotate(q, t_local)
        k = self._rotate(k, t_local)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, l, h)
        out = self.out_proj(out)
        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.ff(x))
        return x


class FourComponentCandidateEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        cand_dim,
        num_components=4,
        hidden_dim=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        hidden_dim_dense=256,
        dropout=0.0,
        logvar_param="clamp",
        logvar_min=-8.0,
        logvar_max=2.0,
        selection_mode="independent",
        num_probes=4,
        use_probe_embedding=False,
        time_encoding="none",
        time_num_freqs=16,
        time_min_hz=1.0,
        time_max_hz=1000.0,
    ):
        super().__init__()
        self.num_components = int(num_components)
        self.logvar_param = str(logvar_param)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.selection_mode = str(selection_mode)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Optional explicit probe identity: an additive per-token embedding keyed
        # by probe id (mirrors the legacy encoder's dropped probe_embedding). The
        # base features only expose probe *angle* via sin/cos(theta); this gives a
        # clean categorical probe signal that per-probe noise / heteroscedastic
        # modeling (#3) can latch onto.
        self.use_probe_embedding = bool(use_probe_embedding)
        if self.use_probe_embedding:
            self.probe_embedding = nn.Embedding(int(num_probes), hidden_dim)
        # ---- continuous-time positional encoding (#4 tier-2) ----
        # All modes key off real (local) sampling time, not token index, because
        # the de-aliasing structure lives in the irregular t_n. 'fourier' adds a
        # multiscale sin/cos(omega_m * t) embedding (the basis in which the BTT
        # phase 2*pi*f*t is linear); 'rope' makes attention relative-in-time.
        self.time_encoding = str(time_encoding)
        if self.time_encoding not in ("none", "fourier", "rope"):
            raise ValueError(f"Unsupported time_encoding={self.time_encoding!r}")
        if self.time_encoding == "fourier":
            freqs = torch.logspace(
                math.log10(float(time_min_hz)),
                math.log10(float(time_max_hz)),
                int(time_num_freqs),
            )
            self.register_buffer("time_omega", 2.0 * math.pi * freqs)  # [M]
            self.time_proj = nn.Linear(2 * int(time_num_freqs), hidden_dim)
        if self.time_encoding == "rope":
            head_dim = hidden_dim // nhead
            rope_freqs = torch.logspace(
                math.log10(float(time_min_hz)),
                math.log10(float(time_max_hz)),
                head_dim // 2,
            )
            rope_omega = 2.0 * math.pi * rope_freqs
            self.rope_layers = nn.ModuleList(
                [
                    RoPETimeEncoderLayer(
                        hidden_dim, nhead, dim_feedforward, dropout, rope_omega.clone()
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers
            )
        self.seq_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim_dense),
            nn.ReLU(),
            nn.Linear(hidden_dim_dense, hidden_dim),
            nn.ReLU(),
        )
        self.component_embedding = nn.Embedding(num_components, hidden_dim)
        self.cand_mlp = nn.Sequential(
            nn.Linear(cand_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.z_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 2)
        # Autoregressive selector: q(z_k | z_<k, y). Picks components in order,
        # carrying a GRU context of previously selected candidates plus an
        # explicit min-distance-to-selected feature so exclusion is easy to learn.
        if self.selection_mode == "autoregressive":
            self.ar_query = nn.Linear(hidden_dim, hidden_dim)
            self.ar_gru = nn.GRUCell(hidden_dim, hidden_dim)
            self.ar_logit = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def ar_logits(self, s, h, cand_freq, prev_idx=None):
        """Conditional selection logits [B,K,J].

        s:[B,H] sequence embedding; h:[B,K,J,H] fused candidate features;
        cand_freq:[B,K,J]; prev_idx:[B,K] teacher-forced selections, or None for
        greedy free-running (eval).
        """
        batch, k_count, top_j, hidden = h.shape
        query = torch.tanh(self.ar_query(s))
        selected_freqs = []
        logits_all = []
        for k in range(k_count):
            h_k = h[:, k]  # [B,J,H]
            q = query.unsqueeze(1).expand(-1, top_j, -1)
            if selected_freqs:
                sf = torch.stack(selected_freqs, dim=1)  # [B, k]
                dist = (cand_freq[:, k].unsqueeze(-1) - sf.unsqueeze(1)).abs().amin(dim=-1, keepdim=True)
            else:
                dist = torch.full((batch, top_j, 1), 1.0e3, device=h.device, dtype=h.dtype)
            feat = torch.cat([h_k, q, torch.log1p(dist)], dim=-1)
            lk = self.ar_logit(feat).squeeze(-1)  # [B,J]
            logits_all.append(lk)
            idx = prev_idx[:, k] if prev_idx is not None else lk.argmax(dim=-1)
            g = h_k.gather(1, idx.view(batch, 1, 1).expand(-1, 1, hidden)).squeeze(1)
            query = self.ar_gru(g, query)
            selected_freqs.append(cand_freq[:, k].gather(1, idx.view(batch, 1)).squeeze(1))
        return torch.stack(logits_all, dim=1)  # [B,K,J]

    def encode_sequence(self, x, mask, probe_ids=None, t=None):
        h = self.input_proj(x)
        if self.use_probe_embedding:
            if probe_ids is None:
                raise ValueError(
                    "use_probe_embedding=True requires probe_ids; if loading a "
                    "candidate cache built before probe_ids existed, delete it so "
                    "it is regenerated with the probe_ids field."
                )
            h = h + self.probe_embedding(probe_ids)
        t_local = None
        if self.time_encoding != "none":
            if t is None:
                raise ValueError(
                    "time_encoding != 'none' requires t (per-token sampling time)"
                )
            # patch-relative seconds: the absolute clock origin is arbitrary and
            # only relative sample timing de-aliases.
            t_local = t - t[:, :1]
        if self.time_encoding == "fourier":
            ang = t_local.unsqueeze(-1) * self.time_omega  # [B, L, M]
            pe = self.time_proj(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))
            h = h + pe
        if self.time_encoding == "rope":
            pad = ~mask
            for layer in self.rope_layers:
                h = layer(h, t_local, key_padding_mask=pad)
        else:
            h = self.transformer_encoder(h, src_key_padding_mask=~mask)
        return self.seq_fc(masked_mean(h, mask))

    def forward(self, x, mask, candidate_features, probe_ids=None, t=None):
        batch, k_count, top_j, _ = candidate_features.shape
        s_vec = self.encode_sequence(x, mask, probe_ids=probe_ids, t=t)
        s = s_vec.view(batch, 1, 1, -1).expand(-1, k_count, top_j, -1)
        c = self.cand_mlp(candidate_features)
        comp_ids = torch.arange(k_count, device=x.device)
        comp = self.component_embedding(comp_ids).view(1, k_count, 1, -1).expand(batch, -1, top_j, -1)
        h = self.fuse(torch.cat([s, c, comp], dim=-1))
        raw = self.delta_head(h)
        raw_logvar = raw[..., 1:]
        if self.logvar_param == "sigmoid":
            # Soft, everywhere-differentiable bound (mirrors Encoder.py's
            # log_rho2 = min + (max-min)*sigmoid). No hard rail to stick to.
            logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(raw_logvar)
        else:
            logvar = raw_logvar.clamp(self.logvar_min, self.logvar_max)
        return {
            "z_logits": self.z_head(h).squeeze(-1),
            "delta_mu": torch.tanh(raw[..., :1]).squeeze(-1),
            "delta_logvar": logvar.squeeze(-1),
            "s": s_vec,
            "h": h,
        }


class JointConstantFrequencyDecoder(nn.Module):
    """Reconstruction head with two amplitude treatments.

    ``point_ls`` (legacy default): amplitudes are a ridge-regularized complex
    least-squares point estimate (VARPRO concentration); the NLL is the masked
    residual energy ``||y - Phi a_hat||^2 / sigma^2``. The ridge is purely
    numerical -- there is no amplitude prior in the objective.

    ``marginal`` (ported from the main project, ``complex_gaussian_marginal_nll``
    + ``centered_complex_gaussian_from_data`` prior): amplitudes are integrated
    out under a zero-mean isotropic complex-Gaussian prior with per-sequence
    variance tau^2 estimated from the data (signal power / K). The NLL is the
    true marginal ``-log p(y | f)`` = quad + logdet, where the prior precision
    1/tau^2 regularizes the Gram and the logdet term is an Occam factor that
    penalizes near-collinear (close-tone) dictionaries -- the regime where
    point-LS amplitude variance blows up.
    """

    def __init__(
        self,
        ridge=1e-5,
        recon_mode="point_ls",
        amp_prior_var_scale=1.0,
        include_log_const=False,
    ):
        super().__init__()
        if recon_mode not in ("point_ls", "marginal"):
            raise ValueError(f"Unsupported recon_mode={recon_mode!r}")
        self.ridge = float(ridge)
        self.recon_mode = recon_mode
        self.amp_prior_var_scale = float(amp_prior_var_scale)
        self.include_log_const = bool(include_log_const)

    def solve_reconstruction(self, y_complex, t, freqs_hz, mask):
        phase = 2.0 * torch.pi * t.unsqueeze(-1) * freqs_hz.unsqueeze(1)
        phi = torch.polar(torch.ones_like(phase), phase)
        w = mask.to(device=y_complex.device, dtype=y_complex.real.dtype).unsqueeze(-1)
        phi_w = phi * w
        y_w = y_complex * w.squeeze(-1)
        phi_h = phi_w.conj().transpose(-2, -1)
        gram = phi_h @ phi_w
        eye = torch.eye(freqs_hz.shape[-1], dtype=gram.dtype, device=gram.device).unsqueeze(0)
        rhs = phi_h @ y_w.unsqueeze(-1)
        amp = torch.linalg.solve(gram + self.ridge * eye, rhs).squeeze(-1)
        y_hat = (phi * amp.unsqueeze(1)).sum(dim=-1)
        return y_hat, amp

    def reconstruction_nll(self, y_complex, t, freqs_hz, mask, noise_var):
        return self.reconstruction_nll_per_sample(
            y_complex,
            t,
            freqs_hz,
            mask,
            noise_var,
        ).mean()

    def reconstruction_nll_per_sample(self, y_complex, t, freqs_hz, mask, noise_var):
        if self.recon_mode == "marginal":
            return self.marginal_nll_per_sample(
                y_complex, t, freqs_hz, mask, noise_var
            )
        y_hat, _ = self.solve_reconstruction(y_complex, t, freqs_hz, mask)
        w = mask.to(device=y_complex.device, dtype=y_complex.real.dtype)
        sqerr = torch.abs((y_complex - y_hat) * w) ** 2
        noise = noise_var.to(device=y_complex.device, dtype=y_complex.real.dtype).clamp_min(1e-12)
        return sqerr.sum(dim=-1) / noise

    def estimate_amp_prior_var(self, y_complex, mask, noise_var, k_count):
        """Empirical-Bayes zero-mean prior variance tau^2 per sequence.

        tau^2 = scale * max(observed_power - sigma^2, eps) / K, i.e. share the
        estimated signal power equally across the K complex amplitudes. With the
        dataset's per-sequence-std normalization observed_power ~ O(1).
        """
        real_dtype = y_complex.real.dtype
        w = mask.to(device=y_complex.device, dtype=real_dtype)
        noise = noise_var.to(device=y_complex.device, dtype=real_dtype).clamp_min(1e-12)
        n_obs = w.sum(dim=-1).clamp_min(1.0)
        obs_power = (torch.abs(y_complex) ** 2 * w).sum(dim=-1) / n_obs
        sig_power = (obs_power - noise).clamp_min(1e-8)
        tau2 = self.amp_prior_var_scale * sig_power / float(max(int(k_count), 1))
        return tau2.clamp_min(1e-10)

    def marginal_nll_per_sample(self, y_complex, t, freqs_hz, mask, noise_var):
        k_count = freqs_hz.shape[-1]
        phase = 2.0 * torch.pi * t.unsqueeze(-1) * freqs_hz.unsqueeze(1)
        phi = torch.polar(torch.ones_like(phase), phase)
        real_dtype = y_complex.real.dtype
        w = mask.to(device=y_complex.device, dtype=real_dtype).unsqueeze(-1)
        phi_w = phi * w
        y_w = y_complex * w.squeeze(-1)
        phi_h = phi_w.conj().transpose(-2, -1)
        gram = phi_h @ phi_w  # [B, K, K]

        sigma2 = noise_var.to(device=y_complex.device, dtype=real_dtype).clamp_min(1e-12)
        tau2 = self.estimate_amp_prior_var(y_complex, mask, noise_var, k_count)
        tau2 = tau2.unsqueeze(-1).expand(-1, k_count)  # [B, K], zero-mean isotropic

        # Zero prior mean => residual is y itself; rhs = Phi^H y (masked).
        rhs_residual = phi_h @ y_w.unsqueeze(-1)  # [B, K, 1]
        precision = torch.diag_embed(1.0 / tau2).to(dtype=gram.dtype)
        system = precision + gram / sigma2.view(-1, 1, 1)  # [B, K, K]

        residual_norm = (torch.abs(y_w) ** 2).sum(dim=-1) / sigma2
        system_solve = torch.linalg.solve(system, rhs_residual)
        correction = torch.real(
            rhs_residual.conj().transpose(-2, -1) @ system_solve
        ).squeeze(-1).squeeze(-1) / sigma2.pow(2)
        quad = (residual_norm - correction).clamp_min(0.0)

        n_obs = w.squeeze(-1).sum(dim=-1).clamp_min(1.0)
        _, logabsdet_system = torch.linalg.slogdet(system)
        logdet = (
            n_obs * torch.log(sigma2)
            + torch.sum(torch.log(tau2), dim=-1)
            + torch.real(logabsdet_system)
        )
        nll = quad + logdet
        if self.include_log_const:
            nll = nll + n_obs * math.log(math.pi)
        return nll


def gather_component_candidates(values, index):
    return values.gather(2, index.unsqueeze(-1)).squeeze(-1)


def gaussian_nll(mu, logvar, target):
    return 0.5 * (((target - mu) ** 2) * torch.exp(-logvar) + logvar)


def soft_frequency_targets(candidate_freq, true_freq, temperature_hz):
    err = torch.abs(candidate_freq - true_freq.unsqueeze(-1))
    return F.softmax(-err / max(float(temperature_hz), 1e-6), dim=-1)


def soft_basin_targets(candidate_freq, radius, true_freq, temperature_hz):
    outside_err = (
        torch.abs(candidate_freq - true_freq.unsqueeze(-1)) - radius
    ).clamp_min(0.0)
    return F.softmax(-outside_err / max(float(temperature_hz), 1e-6), dim=-1)


def cartesian_topk_reconstruction_nll(
    decoder,
    y_complex,
    t,
    refined_all,
    qz,
    mask,
    noise_var,
    topk,
    renormalize=True,
):
    batch, k_count, num_candidates = refined_all.shape
    topk = min(int(topk), int(num_candidates))
    top_prob, top_idx = torch.topk(qz, k=topk, dim=-1)
    top_refined = torch.gather(refined_all, dim=2, index=top_idx)
    component_mass = top_prob.sum(dim=-1).clamp_min(1e-12)
    retained_mass = component_mass.prod(dim=-1)
    if renormalize:
        top_prob = top_prob / component_mass.unsqueeze(-1)
    combo_axes = [torch.arange(topk, device=refined_all.device) for _ in range(k_count)]
    combos = torch.cartesian_prod(*combo_axes)
    combo_count = combos.shape[0]
    combo_freqs = torch.stack(
        [top_refined[:, comp, combos[:, comp]] for comp in range(k_count)],
        dim=-1,
    )
    combo_prob = torch.ones(
        (batch, combo_count),
        dtype=qz.dtype,
        device=qz.device,
    )
    for comp in range(k_count):
        combo_prob = combo_prob * top_prob[:, comp, combos[:, comp]]
    y_flat = (
        y_complex.unsqueeze(1)
        .expand(batch, combo_count, y_complex.shape[-1])
        .reshape(batch * combo_count, y_complex.shape[-1])
    )
    t_flat = (
        t.unsqueeze(1)
        .expand(batch, combo_count, t.shape[-1])
        .reshape(batch * combo_count, t.shape[-1])
    )
    mask_flat = (
        mask.unsqueeze(1)
        .expand(batch, combo_count, mask.shape[-1])
        .reshape(batch * combo_count, mask.shape[-1])
    )
    noise_flat = (
        noise_var.reshape(batch, 1)
        .expand(batch, combo_count)
        .reshape(batch * combo_count)
    )
    nll = decoder.reconstruction_nll_per_sample(
        y_flat,
        t_flat,
        combo_freqs.reshape(batch * combo_count, k_count),
        mask_flat,
        noise_flat,
    ).reshape(batch, combo_count)
    expected_nll = (combo_prob * nll).sum(dim=-1).mean()
    return expected_nll, retained_mass.mean()


def four_component_loss(encoder, decoder, batch, args, device, kl_scale=1.0):
    x = batch["x"].to(device)
    mask = batch["mask"].to(device)
    t = batch["t"].to(device)
    target = batch["target"].to(device)
    y_complex = torch.complex(target[..., 0], target[..., 1])
    cand = batch["candidate_features"].to(device)
    cand_freq = batch["candidate_freq_hz"].to(device)
    radius = batch["local_radius_hz"].to(device).clamp_min(1e-8)
    alias_score = batch["alias_score"].to(device)
    bse_score = batch["bse_score"].to(device)
    ell = batch["ell"].to(device)
    label = batch["label"].to(device)
    true_freq = batch["true_freq_hz"].to(device)
    noise_var = batch["noise_var_norm"].to(device)
    probe_ids = batch["probe_ids"].to(device) if "probe_ids" in batch else None

    out = encoder(x, mask, cand, probe_ids=probe_ids, t=t)
    if getattr(args, "selection_mode", "independent") == "autoregressive":
        # Teacher-force the AR context with the true prior selections in the
        # supervised stage; free-run (own greedy picks) when label-free.
        tf_idx = None if args.unsupervised_elbo_train else label
        out["z_logits"] = encoder.ar_logits(out["s"], out["h"], cand_freq, prev_idx=tf_idx)
    hard_ce = F.cross_entropy(
        out["z_logits"].reshape(-1, out["z_logits"].shape[-1]),
        label.reshape(-1),
    )
    log_qz = F.log_softmax(out["z_logits"], dim=-1)
    qz = torch.exp(log_qz)
    # Soft mutual-exclusion (repulsion): components share one candidate pool and
    # select independently, so closely-coupled tones can collapse onto the same
    # candidate. Penalize the cross-component collision probability
    # sum_{i<j} <q_i, q_j> to push different components onto different candidates.
    if float(args.z_repulsion_weight) > 0.0 and qz.shape[1] >= 2:
        k_count_z = qz.shape[1]
        gram_z = torch.einsum("bkj,blj->bkl", qz, qz)
        off_z = ~torch.eye(k_count_z, dtype=torch.bool, device=qz.device)
        z_repulsion = gram_z[:, off_z].mean()
    else:
        z_repulsion = torch.zeros((), dtype=x.dtype, device=device)
    if args.z_loss_type == "none":
        z_loss = torch.zeros((), dtype=x.dtype, device=device)
    elif args.z_loss_type == "hard_ce":
        z_loss = hard_ce
    elif args.z_loss_type == "soft_freq":
        z_target = soft_frequency_targets(
            candidate_freq=cand_freq,
            true_freq=true_freq,
            temperature_hz=args.z_target_temperature_hz,
        )
        z_loss = -(z_target * log_qz).sum(dim=-1).mean()
    elif args.z_loss_type == "soft_basin":
        z_target = soft_basin_targets(
            candidate_freq=cand_freq,
            radius=radius,
            true_freq=true_freq,
            temperature_hz=args.z_target_temperature_hz,
        )
        z_loss = -(z_target * log_qz).sum(dim=-1).mean()
    elif args.z_loss_type == "soft_bse":
        z_target = F.softmax(
            args.bse_target_beta * torch.nan_to_num(bse_score, nan=0.0),
            dim=-1,
        )
        z_loss = -(z_target * log_qz).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unsupported z_loss_type={args.z_loss_type!r}")
    if args.physics_prior_source == "profile":
        prior_score = ell
        prior_beta = args.profile_prior_beta
    elif args.physics_prior_source == "alias":
        prior_score = alias_score
        prior_beta = args.alias_prior_beta
    elif args.physics_prior_source == "bse":
        prior_score = bse_score
        prior_beta = args.bse_prior_beta
    else:
        raise ValueError(f"Unsupported physics_prior_source={args.physics_prior_source!r}")
    physics_prior = F.softmax(
        prior_beta * torch.nan_to_num(prior_score, nan=0.0),
        dim=-1,
    )
    # ELBO-consistent direction: KL(q_z || prior) = sum_j q_j (log q_j - log p_j).
    # (Previously this was the reverse KL(prior || q_z), which is not the KL that
    # appears in the negative ELBO.) Note prior here is the physics-informed soft
    # prior, not the uniform generative p(z) -- that substitution is a separate
    # modeling choice, left unchanged.
    physics_kl = (
        qz
        * (log_qz - torch.log(physics_prior.clamp_min(1e-8)))
    ).sum(dim=-1).mean()
    delta_kl_all = 0.5 * (
        torch.exp(out["delta_logvar"])
        + out["delta_mu"].pow(2)
        - 1.0
        - out["delta_logvar"]
    )
    cartesian_topk_mass = torch.zeros((), dtype=x.dtype, device=device)
    recon = None
    if args.unsupervised_elbo_train:
        local_nll = torch.zeros((), dtype=x.dtype, device=device)
        local_kl = (qz * delta_kl_all).sum(dim=-1).mean()
        if args.sample_mode == "mean":
            delta_all = out["delta_mu"]
        else:
            eps = torch.randn_like(out["delta_mu"])
            delta_all = (
                out["delta_mu"] + torch.exp(0.5 * out["delta_logvar"]) * eps
            ).clamp(-1.0, 1.0)
        refined_all = (cand_freq + radius * delta_all).clamp_min(1e-6)
        if args.elbo_frequency_source == "posterior_mean":
            freqs = (qz * refined_all).sum(dim=-1).clamp_min(1e-6)
        elif args.elbo_frequency_source == "argmax":
            pred_for_recon = out["z_logits"].argmax(dim=-1)
            freqs = gather_component_candidates(refined_all, pred_for_recon).clamp_min(1e-6)
        elif args.elbo_frequency_source == "cartesian_topk":
            recon, cartesian_topk_mass = cartesian_topk_reconstruction_nll(
                decoder=decoder,
                y_complex=y_complex,
                t=t,
                refined_all=refined_all,
                qz=qz,
                mask=mask,
                noise_var=noise_var,
                topk=args.cartesian_topk,
                renormalize=not args.no_cartesian_topk_renorm,
            )
            freqs = (qz * refined_all).sum(dim=-1).clamp_min(1e-6)
        else:
            raise ValueError(f"Unsupported elbo_frequency_source={args.elbo_frequency_source!r}")
    else:
        sel_cand = gather_component_candidates(cand_freq, label)
        sel_radius = gather_component_candidates(radius, label)
        sel_mu = gather_component_candidates(out["delta_mu"], label)
        sel_logvar = gather_component_candidates(out["delta_logvar"], label)
        target_delta = ((true_freq - sel_cand) / sel_radius).clamp(-1.0, 1.0)
        local_nll = gaussian_nll(sel_mu, sel_logvar, target_delta).mean()
        local_kl = gather_component_candidates(delta_kl_all, label).mean()
        if args.sample_mode == "mean":
            delta = sel_mu
        else:
            eps = torch.randn_like(sel_mu)
            delta = (sel_mu + torch.exp(0.5 * sel_logvar) * eps).clamp(-1.0, 1.0)
        freqs = (sel_cand + sel_radius * delta).clamp_min(1e-6)
    if recon is None:
        recon = decoder.reconstruction_nll(y_complex, t, freqs, mask, noise_var)
    loss = (
        args.recon_weight * recon
        + args.z_loss_weight * z_loss
        + args.physics_prior_weight * physics_kl
        + args.local_nll_weight * local_nll
        + (args.local_kl_weight * float(kl_scale)) * local_kl
        + args.z_repulsion_weight * z_repulsion
    )
    with torch.no_grad():
        pred = out["z_logits"].argmax(dim=-1)
        pred_cand = gather_component_candidates(cand_freq, pred)
        pred_radius = gather_component_candidates(radius, pred)
        pred_mu = gather_component_candidates(out["delta_mu"], pred)
        pred_freq = pred_cand + pred_radius * pred_mu
        raw_err = torch.abs(pred_cand - true_freq)
        pred_basin_err = (raw_err - pred_radius).clamp_min(0.0)
        err = torch.abs(pred_freq - true_freq)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "z_loss": float(z_loss.detach().cpu()),
        "hard_ce": float(hard_ce.detach().cpu()),
        "physics_kl": float(physics_kl.detach().cpu()),
        "local_nll": float(local_nll.detach().cpu()),
        "local_kl": float(local_kl.detach().cpu()),
        "z_repulsion": float(z_repulsion.detach().cpu()),
        "kl_scale": float(kl_scale),
        "delta_logvar_mean": float(out["delta_logvar"].detach().mean().cpu()),
        "cartesian_topk_mass": float(cartesian_topk_mass.detach().cpu()),
        "component_top1": float((pred == label).float().mean().detach().cpu()),
        "component_basin_hit": float(
            (pred_basin_err <= args.coverage_tol_hz).float().mean().detach().cpu()
        ),
        "freq_mae_hz": float(err.mean().detach().cpu()),
        "raw_freq_mae_hz": float(raw_err.mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate(encoder, decoder, loader, device, coverage_tol_hz=1.0):
    encoder.eval()
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        t = batch["t"].to(device)
        cand = batch["candidate_features"].to(device)
        cand_freq = batch["candidate_freq_hz"].to(device)
        radius = batch["local_radius_hz"].to(device)
        label = batch["label"].to(device)
        true_freq = batch["true_freq_hz"].to(device)
        probe_ids = batch["probe_ids"].to(device) if "probe_ids" in batch else None
        out = encoder(x, mask, cand, probe_ids=probe_ids, t=t)
        if getattr(encoder, "selection_mode", "independent") == "autoregressive":
            out["z_logits"] = encoder.ar_logits(out["s"], out["h"], cand_freq, prev_idx=None)
        pred = out["z_logits"].argmax(dim=-1)
        pred_cand = gather_component_candidates(cand_freq, pred)
        pred_radius = gather_component_candidates(radius, pred)
        pred_mu = gather_component_candidates(out["delta_mu"], pred)
        pred_freq = pred_cand + pred_radius * pred_mu
        qz = F.softmax(out["z_logits"], dim=-1)
        refined_all = cand_freq + radius * out["delta_mu"]
        posterior_mean_freq = (qz * refined_all).sum(dim=-1)
        err = torch.abs(pred_freq - true_freq)
        posterior_mean_err = torch.abs(posterior_mean_freq - true_freq)
        raw_err = torch.abs(pred_cand - true_freq)
        pred_basin_hit = (
            (raw_err - pred_radius).clamp_min(0.0) <= coverage_tol_hz
        )
        # Merge detection for close-spacing studies: smallest pairwise gap among
        # the predicted refined frequencies, and whether all selected candidate
        # centers are pairwise distinct (a collapse onto one tone => not distinct).
        k_count = pred_freq.shape[1]
        if k_count >= 2:
            off_diag = ~torch.eye(k_count, device=pred_freq.device, dtype=torch.bool)
            pf_gap = (pred_freq.unsqueeze(1) - pred_freq.unsqueeze(2)).abs()
            min_pair_gap = pf_gap[:, off_diag].view(pred_freq.shape[0], -1).amin(dim=-1)
            cc_gap = (pred_cand.unsqueeze(1) - pred_cand.unsqueeze(2)).abs()
            min_cc_gap = cc_gap[:, off_diag].view(pred_freq.shape[0], -1).amin(dim=-1)
            distinct_centers = (min_cc_gap > 1e-6).float()
        else:
            min_pair_gap = torch.full((pred_freq.shape[0],), float("nan"))
            distinct_centers = torch.ones(pred_freq.shape[0])
        rows.append({
            "component_top1": (pred == label).float().cpu(),
            "sequence_top1": torch.all(pred == label, dim=1).float().cpu(),
            "component_basin_hit": pred_basin_hit.float().cpu(),
            "sequence_basin_hit": torch.all(pred_basin_hit, dim=1).float().cpu(),
            "err": err.cpu(),
            "posterior_mean_err": posterior_mean_err.cpu(),
            "raw_err": raw_err.cpu(),
            "min_pair_gap": min_pair_gap.cpu(),
            "distinct_centers": distinct_centers.cpu(),
        })
    comp = torch.cat([r["component_top1"].reshape(-1) for r in rows])
    seq = torch.cat([r["sequence_top1"] for r in rows])
    comp_basin_hit = torch.cat([r["component_basin_hit"].reshape(-1) for r in rows])
    seq_basin_hit = torch.cat([r["sequence_basin_hit"] for r in rows])
    err = torch.cat([r["err"].reshape(-1) for r in rows])
    posterior_mean_err = torch.cat([r["posterior_mean_err"].reshape(-1) for r in rows])
    raw_err = torch.cat([r["raw_err"].reshape(-1) for r in rows])
    min_pair_gap = torch.cat([r["min_pair_gap"] for r in rows])
    distinct_centers = torch.cat([r["distinct_centers"] for r in rows])
    return {
        "component_top1": float(comp.mean()),
        "sequence_top1": float(seq.mean()),
        "component_basin_hit": float(comp_basin_hit.mean()),
        "sequence_basin_hit": float(seq_basin_hit.mean()),
        "freq_mae_hz": float(err.mean()),
        "freq_rmse_hz": float(torch.sqrt(torch.mean(err.pow(2)))),
        "posterior_mean_freq_mae_hz": float(posterior_mean_err.mean()),
        "posterior_mean_freq_rmse_hz": float(torch.sqrt(torch.mean(posterior_mean_err.pow(2)))),
        "raw_freq_mae_hz": float(raw_err.mean()),
        "distinct_center_rate": float(distinct_centers.mean()),
        "pred_min_pair_gap_hz": float(torch.nanmean(min_pair_gap)),
    }


@torch.no_grad()
def audit_candidate_quality(loader, args, device):
    rows = []
    for batch in loader:
        cand_freq = batch["candidate_freq_hz"].to(device)
        radius = batch["local_radius_hz"].to(device)
        true_freq = batch["true_freq_hz"].to(device)
        ell = batch["ell"].to(device)
        alias_score = batch["alias_score"].to(device)
        bse_score = batch["bse_score"].to(device)
        err = torch.abs(cand_freq - true_freq.unsqueeze(-1))
        oracle_err = err.min(dim=-1).values
        interval_err = (err - radius).clamp_min(0.0)
        basin_oracle_err = interval_err.min(dim=-1).values
        profile_idx = ell.argmax(dim=-1)
        profile_err = gather_component_candidates(err, profile_idx)
        profile_basin_err = gather_component_candidates(interval_err, profile_idx)
        alias_idx = alias_score.argmax(dim=-1)
        alias_err = gather_component_candidates(err, alias_idx)
        alias_basin_err = gather_component_candidates(interval_err, alias_idx)
        bse_idx = bse_score.argmax(dim=-1)
        bse_err = gather_component_candidates(err, bse_idx)
        bse_basin_err = gather_component_candidates(interval_err, bse_idx)
        label = batch["label"].to(device)
        label_err = gather_component_candidates(err, label)
        label_basin_err = gather_component_candidates(interval_err, label)
        rows.append(
            {
                "oracle_err": oracle_err.cpu(),
                "basin_oracle_err": basin_oracle_err.cpu(),
                "profile_err": profile_err.cpu(),
                "profile_basin_err": profile_basin_err.cpu(),
                "alias_err": alias_err.cpu(),
                "alias_basin_err": alias_basin_err.cpu(),
                "bse_err": bse_err.cpu(),
                "bse_basin_err": bse_basin_err.cpu(),
                "label_err": label_err.cpu(),
                "label_basin_err": label_basin_err.cpu(),
                "coverage": (oracle_err <= args.coverage_tol_hz).float().cpu(),
                "basin_coverage": (
                    basin_oracle_err <= args.coverage_tol_hz
                ).float().cpu(),
                "sequence_coverage": torch.all(
                    oracle_err <= args.coverage_tol_hz,
                    dim=1,
                ).float().cpu(),
                "sequence_basin_coverage": torch.all(
                    basin_oracle_err <= args.coverage_tol_hz,
                    dim=1,
                ).float().cpu(),
                "per_component_basin_coverage": (
                    basin_oracle_err <= args.coverage_tol_hz
                ).float().cpu(),
            }
        )
    cat = {key: torch.cat([row[key].reshape(-1) for row in rows]) for key in rows[0]}
    per_component_basin_coverage = torch.cat(
        [row["per_component_basin_coverage"] for row in rows],
        dim=0,
    ).mean(dim=0)
    metrics = {
        "oracle_candidate_mae_hz": float(cat["oracle_err"].mean()),
        "oracle_candidate_rmse_hz": float(torch.sqrt(torch.mean(cat["oracle_err"].pow(2)))),
        "candidate_coverage_at_tol": float(cat["coverage"].mean()),
        "sequence_candidate_coverage_at_tol": float(cat["sequence_coverage"].mean()),
        "basin_oracle_mae_hz": float(cat["basin_oracle_err"].mean()),
        "basin_oracle_rmse_hz": float(torch.sqrt(torch.mean(cat["basin_oracle_err"].pow(2)))),
        "basin_coverage_at_tol": float(cat["basin_coverage"].mean()),
        "sequence_basin_coverage_at_tol": float(cat["sequence_basin_coverage"].mean()),
        "per_component_basin_coverage_at_tol": [
            float(value) for value in per_component_basin_coverage
        ],
        "profile_top1_mae_hz": float(cat["profile_err"].mean()),
        "profile_top1_basin_mae_hz": float(cat["profile_basin_err"].mean()),
        "alias_top1_mae_hz": float(cat["alias_err"].mean()),
        "alias_top1_basin_mae_hz": float(cat["alias_basin_err"].mean()),
        "bse_top1_mae_hz": float(cat["bse_err"].mean()),
        "bse_top1_basin_mae_hz": float(cat["bse_basin_err"].mean()),
        "label_mae_hz": float(cat["label_err"].mean()),
        "label_basin_mae_hz": float(cat["label_basin_err"].mean()),
    }
    return metrics


def warm_start_encoder_from_checkpoint(encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    current = encoder.state_dict()
    mapped = {}
    skipped = {}
    for key, value in state.items():
        if not key.startswith("encoder."):
            continue
        legacy_key = key[len("encoder.") :]
        target_key = legacy_key
        if legacy_key == "_fc.weight":
            target_key = "seq_fc.0.weight"
        elif legacy_key == "_fc.bias":
            target_key = "seq_fc.0.bias"
        elif legacy_key.startswith((
            "freq_",
            "f_center",
            "f_band",
            "_fc_f_mu",
            "_fc_f_logvar",
            "probe_embedding",
        )):
            skipped[legacy_key] = "not used"
            continue
        if target_key not in current:
            skipped[legacy_key] = "no target"
            continue
        if tuple(current[target_key].shape) != tuple(value.shape):
            skipped[legacy_key] = f"{tuple(value.shape)} -> {tuple(current[target_key].shape)}"
            continue
        mapped[target_key] = value
    current.update(mapped)
    encoder.load_state_dict(current)
    return {"checkpoint": str(checkpoint_path), "loaded_keys": sorted(mapped), "skipped": skipped}


def load_encoder_state_checkpoint(encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("encoder.") for key in state):
        state = {
            key[len("encoder.") :]: value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
    current = encoder.state_dict()
    loaded = {}
    skipped = {}
    for key, value in state.items():
        if key not in current:
            skipped[key] = "no target"
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped[key] = f"{tuple(value.shape)} -> {tuple(current[key].shape)}"
            continue
        loaded[key] = value
    current.update(loaded)
    encoder.load_state_dict(current)
    return {"checkpoint": str(checkpoint_path), "loaded_keys": sorted(loaded), "skipped": skipped}


def configure_z_only_training(encoder):
    frozen_prefixes = (
        "input_proj.",
        "probe_embedding.",
        "time_proj.",
        "transformer_encoder.",
        "rope_layers.",
        "seq_fc.",
        "delta_head.",
    )
    for name, param in encoder.named_parameters():
        param.requires_grad = not name.startswith(frozen_prefixes)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    trainable_names = [
        name for name, param in encoder.named_parameters() if param.requires_grad
    ]
    return {
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_tensors": trainable_names,
    }


def configure_unsupervised_elbo_training(encoder, args):
    frozen_prefixes = []
    if args.freeze_backbone_for_elbo:
        frozen_prefixes.extend([
            "input_proj.",
            "transformer_encoder.",
            "seq_fc.",
        ])
    if args.freeze_z_path_for_elbo:
        frozen_prefixes.extend([
            "component_embedding.",
            "cand_mlp.",
            "fuse.",
            "z_head.",
        ])
    for name, param in encoder.named_parameters():
        param.requires_grad = not any(name.startswith(prefix) for prefix in frozen_prefixes)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    trainable_names = [
        name for name, param in encoder.named_parameters() if param.requires_grad
    ]
    return {
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_tensors": trainable_names,
    }


# Argument names that determine the (model-independent) candidate tensors. Two
# runs that agree on all of these produce byte-identical candidates, so the
# cache can be reused across training configs that only differ in optimizer /
# loss / epoch settings.
CANDIDATE_CACHE_KEYS = (
    "snr_db",
    "num_cycles",
    "speed_profile",
    "speed_change",
    "controlled_spacing_hz",
    "num_components",
    "center_band_hz",
    "spacing_jitter_hz",
    "search_relative_half_band",
    "grid_size",
    "top_j",
    "candidate_support_mode",
    "candidate_mode",
    "candidate_search_mode",
    "global_fmin_hz",
    "global_fmax_hz",
    "global_bins",
    "basin_peak_count",
    "basin_round_candidates",
    "basin_greedy_rounds",
    "basin_alias_corr_threshold",
    "basin_alias_max_residual_drop_ratio",
    "basin_alias_members_per_family",
    "basin_candidate_min_gap_bins",
    "basin_half_width_bins",
    "basin_min_half_width_hz",
    "basin_refine_bins",
    "basin_refine_passes",
    "local_radius_hz",
    "profile_ridge",
    "alias_order_max",
    "alias_obs_min_hz",
    "alias_obs_max_hz",
    "bse_time_bins",
    "bse_axis_bins",
    "bse_window_divisor",
    "bse_k_min",
    "bse_k_max",
    "bse_sigma_hz",
    "bse_sigma_scale",
    "include_local_time",
)


def candidate_cache_signature(args, split, n_samples, seed):
    payload = {key: getattr(args, key) for key in CANDIDATE_CACHE_KEYS}
    payload.update({"split": split, "n_samples": n_samples, "seed": seed})
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:16]


class CachedCandidateDataset(Dataset):
    """Materialize a candidate dataset once and serve from stacked tensors.

    Candidate generation (greedy multi-tone profile-LS, alias grouping, optional
    BSE scoring) is deterministic given the data seed and the candidate-search
    arguments, but it runs in NumPy inside ``__getitem__``. Regenerating it every
    epoch starves the GPU, so this wrapper generates the full split exactly once
    (optionally persisting to disk) and then indexes pre-stacked tensors.
    """

    def __init__(self, base, cache_path=None):
        self.num_components = getattr(base, "num_components", None)
        if cache_path is not None and os.path.exists(cache_path):
            self.tensors = torch.load(cache_path)
            print(f"[cache] loaded {len(self)} samples <- {cache_path}")
        else:
            t0 = time.time()
            self.tensors = self._materialize(base)
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                tmp = f"{cache_path}.tmp"
                torch.save(self.tensors, tmp)
                os.replace(tmp, cache_path)
                dst = f" -> {cache_path}"
            else:
                dst = " (in-memory)"
            print(f"[cache] generated {len(self)} samples in {time.time() - t0:.1f}s{dst}")

    @staticmethod
    def _materialize(base):
        buffers = None
        for idx in range(len(base)):
            item = base[idx]
            if buffers is None:
                buffers = {key: [] for key in item}
            for key, value in item.items():
                buffers[key].append(value)
        return {key: torch.stack(values, dim=0) for key, values in buffers.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).shape[0]

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.tensors.items()}


def build_loaders(args):
    specs = [
        ("train", args.train_samples, args.seed + 1),
        ("val", args.val_samples, args.seed + 2),
        ("test", args.test_samples, args.seed + 3),
    ]
    datasets = {}
    for split, n_samples, seed in specs:
        base = LegacyFourComponentCandidateDataset(split, n_samples, seed, args)
        if args.candidate_cache_dir:
            sig = candidate_cache_signature(args, split, n_samples, seed)
            cache_path = os.path.join(args.candidate_cache_dir, f"{split}_{n_samples}_{sig}.pt")
            datasets[split] = CachedCandidateDataset(base, cache_path)
        elif args.precompute_cache:
            datasets[split] = CachedCandidateDataset(base, None)
        else:
            datasets[split] = base
    train_ds = datasets["train"]
    return (
        train_ds,
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(datasets["val"], batch_size=args.batch_size),
        DataLoader(datasets["test"], batch_size=args.batch_size),
    )


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds, train_loader, val_loader, test_loader = build_loaders(args)
    if args.precompute_only:
        print("[precompute] candidate caches ready; exiting before training")
        return
    audit_device = torch.device("cpu")
    candidate_audit = {
        "train": audit_candidate_quality(train_loader, args, audit_device),
        "val": audit_candidate_quality(val_loader, args, audit_device),
        "test": audit_candidate_quality(test_loader, args, audit_device),
    }
    print(
        "[audit] "
        f"test_oracle_mae={candidate_audit['test']['oracle_candidate_mae_hz']:.4f} Hz "
        f"test_coverage@{args.coverage_tol_hz:g}Hz="
        f"{candidate_audit['test']['candidate_coverage_at_tol']:.3f} "
        f"test_basin_oracle_mae={candidate_audit['test']['basin_oracle_mae_hz']:.4f} Hz "
        f"test_basin_coverage@{args.coverage_tol_hz:g}Hz="
        f"{candidate_audit['test']['basin_coverage_at_tol']:.3f} "
        f"profile_top1_mae={candidate_audit['test']['profile_top1_mae_hz']:.4f} Hz "
        f"alias_top1_mae={candidate_audit['test']['alias_top1_mae_hz']:.4f} Hz "
        f"bse_top1_mae={candidate_audit['test']['bse_top1_mae_hz']:.4f} Hz"
    )
    if args.audit_only:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"candidate_audit": candidate_audit, "args": vars(args)}
        (out_dir / "candidate_audit.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"[audit] wrote -> {out_dir / 'candidate_audit.json'}")
        return
    encoder = FourComponentCandidateEncoder(
        input_dim=8 if args.include_local_time else 6,
        cand_dim=6,
        num_components=train_ds.num_components,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        hidden_dim_dense=args.hidden_dim_dense,
        dropout=args.dropout,
        logvar_param=args.delta_logvar_param,
        logvar_min=args.delta_logvar_min,
        logvar_max=args.delta_logvar_max,
        selection_mode=args.selection_mode,
        num_probes=int(CONFIG["data"]["num_probes"]),
        use_probe_embedding=args.use_probe_embedding,
        time_encoding=args.time_encoding,
        time_num_freqs=args.time_num_freqs,
        time_min_hz=args.time_min_hz,
        time_max_hz=args.time_max_hz,
    ).to(device)
    decoder = JointConstantFrequencyDecoder(
        ridge=args.profile_ridge,
        recon_mode=args.recon_mode,
        amp_prior_var_scale=args.amp_prior_var_scale,
        include_log_const=args.marginal_include_log_const,
    ).to(device)
    warm_start = None
    if args.warm_start_checkpoint:
        warm_start = warm_start_encoder_from_checkpoint(encoder, args.warm_start_checkpoint)
        print(f"[warm-start] loaded {len(warm_start['loaded_keys'])} compatible keys")
    init_state = None
    if args.init_encoder_checkpoint:
        init_state = load_encoder_state_checkpoint(encoder, args.init_encoder_checkpoint)
        print(f"[init-encoder] loaded {len(init_state['loaded_keys'])} compatible keys")
    if args.eval_only:
        test_stats = evaluate(encoder, decoder, test_loader, device, args.coverage_tol_hz)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "eval_only": True,
            "test": test_stats,
            "candidate_audit": candidate_audit,
            "args": vars(args),
            "init_state": init_state,
        }
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[eval-only] speed_profile={args.speed_profile} speed_change={args.speed_change:g} -> {out_dir}")
        print(json.dumps(test_stats, indent=2))
        return
    trainable = {
        "trainable_parameters": sum(p.numel() for p in encoder.parameters()),
        "frozen_parameters": 0,
        "trainable_tensors": [
            name for name, param in encoder.named_parameters() if param.requires_grad
        ],
    }
    if args.z_only_train:
        args.recon_weight = 0.0
        args.local_nll_weight = 0.0
        args.local_kl_weight = 0.0
        trainable = configure_z_only_training(encoder)
        print(
            "[z-only] "
            f"trainable={trainable['trainable_parameters']} "
            f"frozen={trainable['frozen_parameters']}"
        )
    if args.unsupervised_elbo_train:
        args.z_loss_type = "none"
        args.z_loss_weight = 0.0
        args.local_nll_weight = 0.0
        trainable = configure_unsupervised_elbo_training(encoder, args)
        print(
            "[unsup-elbo] "
            f"freq_source={args.elbo_frequency_source} "
            f"trainable={trainable['trainable_parameters']} "
            f"frozen={trainable['frozen_parameters']}"
        )
    params = [param for param in encoder.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_state = None
    best_val = -float("inf") if args.z_only_train else float("inf")
    start = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        agg = {}
        for batch in train_loader:
            if args.kl_anneal_steps > 0:
                frac = min(1.0, global_step / float(args.kl_anneal_steps))
                if getattr(args, "kl_anneal_mode", "up") == "down":
                    # high-to-low: start with strong KL (stabilize the local posterior
                    # shape near N(0,1)), then relax so reconstruction drives delta to
                    # actually refine inside the basin. Floors at kl_anneal_floor.
                    kl_scale = 1.0 - (1.0 - float(args.kl_anneal_floor)) * frac
                else:
                    kl_scale = frac
            else:
                kl_scale = 1.0
            global_step += 1
            loss, stats = four_component_loss(
                encoder, decoder, batch, args, device, kl_scale=kl_scale
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            for key, value in stats.items():
                agg[key] = agg.get(key, 0.0) + value
        train_stats = {k: v / max(len(train_loader), 1) for k, v in agg.items()}
        val_stats = evaluate(encoder, decoder, val_loader, device, args.coverage_tol_hz)
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        })
        if args.z_only_train:
            improved = val_stats["component_basin_hit"] > best_val
            score = val_stats["component_basin_hit"]
        elif args.unsupervised_elbo_train and args.elbo_frequency_source == "posterior_mean":
            improved = val_stats["posterior_mean_freq_mae_hz"] < best_val
            score = val_stats["posterior_mean_freq_mae_hz"]
        else:
            improved = val_stats["freq_mae_hz"] < best_val
            score = val_stats["freq_mae_hz"]
        if improved:
            best_val = score
            best_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
        if args.progress_every and (
            epoch == 1 or epoch == args.epochs or epoch % args.progress_every == 0
        ):
            print(
                f"[four-vi] epoch={epoch} loss={train_stats['loss']:.4f} "
                f"val_comp_top1={val_stats['component_top1']:.3f} "
                f"val_basin_hit={val_stats['component_basin_hit']:.3f} "
                f"val_mae={val_stats['freq_mae_hz']:.4f} Hz "
                f"val_post_mean_mae={val_stats['posterior_mean_freq_mae_hz']:.4f} Hz "
                f"retained_mass={train_stats['cartesian_topk_mass']:.4f}"
            )
    if best_state is not None:
        encoder.load_state_dict(best_state)
    test_stats = evaluate(encoder, decoder, test_loader, device, args.coverage_tol_hz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out_dir / "four_component_candidate_encoder.pt")
    payload = {
        "runtime_seconds": time.time() - start,
        "history": history,
        "test": test_stats,
        "candidate_audit": candidate_audit,
        "args": vars(args),
        "warm_start": warm_start,
        "init_state": init_state,
        "trainable": trainable,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(
        "# Legacy Four-Component Candidate VI\n\n"
        "| Component Top-1 | Sequence Top-1 | Component Basin Hit | Sequence Basin Hit | Freq MAE Hz | Posterior Mean MAE Hz | Freq RMSE Hz | Raw Freq MAE Hz |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {test_stats['component_top1']:.4f} | {test_stats['sequence_top1']:.4f} | "
        f"{test_stats['component_basin_hit']:.4f} | {test_stats['sequence_basin_hit']:.4f} | "
        f"{test_stats['freq_mae_hz']:.4f} | {test_stats['posterior_mean_freq_mae_hz']:.4f} | "
        f"{test_stats['freq_rmse_hz']:.4f} | "
        f"{test_stats['raw_freq_mae_hz']:.4f} |\n",
        encoding="utf-8",
    )
    print(f"[four-vi] done -> {out_dir}")
    print(json.dumps(test_stats, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Four-component legacy synthetic candidate validation")
    p.add_argument("--out_dir", default="artifacts/legacy_four_component_candidate_vi/run")
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--train_samples", type=int, default=256)
    p.add_argument("--val_samples", type=int, default=64)
    p.add_argument("--test_samples", type=int, default=64)
    p.add_argument("--num_cycles", type=int, default=4)
    p.add_argument("--snr_db", type=float, default=20.0)
    p.add_argument(
        "--speed_profile",
        choices=["random", "constant", "linear_up", "linear_down"],
        default="random",
        help="Rotor-speed trajectory over revolutions (frequency stays constant). 'random' = legacy jitter.",
    )
    p.add_argument(
        "--speed_change",
        type=float,
        default=0.0,
        help="Relative-speed ramp magnitude for linear_up/linear_down (e.g. 0.1 = +/-10%).",
    )
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Load --init_encoder_checkpoint and only run audit+evaluate (no training); for zero-shot OOD tests.",
    )
    p.add_argument(
        "--controlled_spacing_hz",
        type=float,
        default=0.0,
        help="If >0, replace config frequency centers with a fixed-gap comb of this spacing (close-spacing resolution study).",
    )
    p.add_argument(
        "--num_components",
        type=int,
        default=2,
        help="Number of comb tones when --controlled_spacing_hz>0 (ignored otherwise; config decides).",
    )
    p.add_argument("--center_band_hz", type=float, default=500.0)
    p.add_argument("--spacing_jitter_hz", type=float, default=0.0)
    p.add_argument("--search_relative_half_band", type=float, default=0.15)
    p.add_argument("--grid_size", type=int, default=160)
    p.add_argument("--top_j", type=int, default=24)
    p.add_argument("--candidate_support_mode", choices=["band", "global_fmax", "basin"], default="basin")
    p.add_argument("--global_fmin_hz", type=float, default=1.0)
    p.add_argument("--global_fmax_hz", type=float, default=1000.0)
    p.add_argument("--global_bins", type=int, default=512)
    p.add_argument("--basin_peak_count", type=int, default=80)
    p.add_argument("--basin_round_candidates", type=int, default=8)
    p.add_argument("--basin_greedy_rounds", type=int, default=4)
    p.add_argument("--basin_alias_corr_threshold", type=float, default=0.95)
    p.add_argument("--basin_alias_max_residual_drop_ratio", type=float, default=0.05)
    p.add_argument("--basin_alias_members_per_family", type=int, default=3)
    p.add_argument("--basin_candidate_min_gap_bins", type=float, default=2.0)
    p.add_argument("--basin_half_width_bins", type=float, default=1.0)
    p.add_argument("--basin_min_half_width_hz", type=float, default=20.0)
    p.add_argument("--basin_refine_bins", type=int, default=9)
    p.add_argument("--basin_refine_passes", type=int, default=1)
    p.add_argument("--local_radius_hz", type=float, default=0.25)
    p.add_argument("--candidate_mode", choices=["profile_ls", "alias_hybrid", "bse_hybrid"], default="alias_hybrid")
    p.add_argument(
        "--candidate_search_mode",
        choices=["serial", "serial_global", "independent"],
        default="serial_global",
    )
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--hidden_dim_dense", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--profile_ridge", type=float, default=1e-5)
    p.add_argument(
        "--recon_mode",
        choices=["point_ls", "marginal"],
        default="point_ls",
        help="point_ls: ridge LS amplitude point estimate (legacy). "
        "marginal: integrate out a zero-mean complex-Gaussian amplitude prior "
        "(complex_gaussian_marginal_nll, ported from the main project).",
    )
    p.add_argument(
        "--amp_prior_var_scale",
        type=float,
        default=1.0,
        help="Empirical-Bayes scale for the marginal amplitude prior variance "
        "tau^2 = scale * (signal_power / K).",
    )
    p.add_argument("--marginal_include_log_const", action="store_true")
    p.add_argument(
        "--include_local_time",
        action="store_true",
        help="#4 tier-1: append two time channels to the encoder input "
        "(input_dim 6->8): normalized local time t_local=(t-t0)/(t_max-t0) "
        "and normalized sampling interval dt_norm=(t_n-t_{n-1})/median(dt). "
        "Off by default for backward compat with 6-dim ckpts.",
    )
    p.add_argument(
        "--use_probe_embedding",
        action="store_true",
        help="Add an explicit additive per-token probe embedding (nn.Embedding "
        "keyed by probe id) to the encoder input, mirroring the legacy encoder. "
        "Gives a clean categorical probe signal beyond the implicit probe angle "
        "in sin/cos(theta). Off by default for backward compat with ckpts that "
        "lack the probe_embedding weights.",
    )
    p.add_argument(
        "--time_encoding",
        choices=["none", "fourier", "rope"],
        default=CONFIG["model"].get("time_encoding", "none"),
        help="#4 tier-2 continuous-time positional encoding keyed by real t_n "
        "(not token index). 'fourier': additive multiscale sin/cos(omega_m*t) "
        "embedding; 'rope': rotary attention relative-in-time. Default 'none'.",
    )
    p.add_argument(
        "--time_num_freqs",
        type=int,
        default=CONFIG["model"].get("time_num_freqs", 16),
        help="Number of log-spaced frequencies in the fourier time-encoding bank "
        "(rope uses head_dim//2 internally).",
    )
    p.add_argument(
        "--time_min_hz",
        type=float,
        default=CONFIG["model"].get("time_min_hz", 1.0),
        help="Lowest frequency (Hz) of the time-encoding bank; ~1/window.",
    )
    p.add_argument(
        "--time_max_hz",
        type=float,
        default=CONFIG["model"].get("time_max_hz", 1000.0),
        help="Highest frequency (Hz) of the time-encoding bank; ~signal f_max.",
    )
    p.add_argument("--sample_mode", choices=["mean", "sample"], default="sample")
    p.add_argument("--recon_weight", type=float, default=0.01)
    p.add_argument("--z_only_train", action="store_true")
    p.add_argument("--unsupervised_elbo_train", action="store_true")
    p.add_argument(
        "--elbo_frequency_source",
        choices=["posterior_mean", "argmax", "cartesian_topk"],
        default="posterior_mean",
    )
    p.add_argument("--cartesian_topk", type=int, default=3)
    p.add_argument("--no_cartesian_topk_renorm", action="store_true")
    p.add_argument("--freeze_backbone_for_elbo", action="store_true")
    p.add_argument("--freeze_z_path_for_elbo", action="store_true")
    p.add_argument("--init_encoder_checkpoint", default="")
    p.add_argument("--z_loss_type", choices=["none", "hard_ce", "soft_freq", "soft_basin", "soft_bse"], default="soft_basin")
    p.add_argument("--z_loss_weight", type=float, default=1.0)
    p.add_argument(
        "--selection_mode",
        choices=["independent", "autoregressive"],
        default="independent",
        help="independent = per-component q(z_k|y); autoregressive = q(z_k|z_<k,y) for built-in mutual exclusion.",
    )
    p.add_argument(
        "--z_repulsion_weight",
        type=float,
        default=0.0,
        help="If >0, penalize cross-component selection collision sum_{i<j}<q_i,q_j> to enforce distinct tones.",
    )
    p.add_argument("--z_target_temperature_hz", type=float, default=1.0)
    p.add_argument("--physics_prior_source", choices=["profile", "alias", "bse"], default="bse")
    p.add_argument("--physics_prior_weight", type=float, default=0.0)
    p.add_argument("--profile_prior_beta", type=float, default=1.0)
    p.add_argument("--alias_prior_beta", type=float, default=1.0)
    p.add_argument("--bse_prior_beta", type=float, default=1.0)
    p.add_argument("--bse_target_beta", type=float, default=1.0)
    p.add_argument("--local_nll_weight", type=float, default=1.0)
    p.add_argument("--local_kl_weight", type=float, default=0.01)
    p.add_argument(
        "--delta_logvar_param",
        choices=["clamp", "sigmoid"],
        default="clamp",
        help="How delta_logvar is bounded: hard clamp (default) or soft sigmoid bound.",
    )
    p.add_argument("--delta_logvar_min", type=float, default=-8.0)
    p.add_argument("--delta_logvar_max", type=float, default=2.0)
    p.add_argument(
        "--kl_anneal_steps",
        type=int,
        default=0,
        help="If >0, ramp the delta-KL coefficient over this many optimizer steps.",
    )
    p.add_argument(
        "--kl_anneal_mode",
        choices=["up", "down"],
        default="up",
        help="'up': 0->1 (default). 'down': 1->kl_anneal_floor (strong KL first to "
        "stabilize the local posterior, then relax so reconstruction refines delta).",
    )
    p.add_argument("--kl_anneal_floor", type=float, default=0.05,
                   help="Final delta-KL scale for kl_anneal_mode=down.")
    p.add_argument("--warm_start_checkpoint", default="")
    p.add_argument("--alias_order_max", type=int, default=4)
    p.add_argument("--alias_obs_min_hz", type=float, default=0.0)
    p.add_argument("--alias_obs_max_hz", type=float, default=2000.0)
    p.add_argument("--bse_time_bins", type=int, default=5)
    p.add_argument("--bse_axis_bins", type=int, default=256)
    p.add_argument("--bse_window_divisor", type=float, default=3.0)
    p.add_argument("--bse_k_min", type=int, default=-8)
    p.add_argument("--bse_k_max", type=int, default=8)
    p.add_argument("--bse_sigma_hz", type=float, default=2.0)
    p.add_argument("--bse_sigma_scale", type=float, default=0.1)
    p.add_argument("--coverage_tol_hz", type=float, default=1.0)
    p.add_argument("--audit_only", action="store_true")
    p.add_argument(
        "--candidate_cache_dir",
        default="",
        help="If set, generate candidate tensors once per split and persist/reuse them here.",
    )
    p.add_argument(
        "--precompute_cache",
        action="store_true",
        help="Materialize candidates in memory once even without an on-disk cache dir.",
    )
    p.add_argument(
        "--precompute_only",
        action="store_true",
        help="Build/persist candidate caches for all splits, then exit before training.",
    )
    p.add_argument("--device", default="")
    p.add_argument("--progress_every", type=int, default=2)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
