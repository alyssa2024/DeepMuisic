import argparse
import copy
import csv
import json
import os

import numpy as np
import torch

from config import CONFIG
import main as train_main
from run_profile_likelihood_failure_scan import (
    build_scan_tensors,
    estimated_frequency_from_metrics,
    find_local_minima,
    scan_profile_likelihood,
)
from synthesis_dataset import compute_frequency_support


BASE_CONFIG = copy.deepcopy(CONFIG)
DEFAULT_PROBE_COUNTS = [6]
DEFAULT_HALF_BANDS_HZ = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0]
DEFAULT_SEEDS = list(range(10))
DEFAULT_INIT_MODES = ["center", "oracle"]
DEFAULT_TRUE_FREQUENCY_POSITION = 0.1
FIXED_NUM_HARMONICS = 1
FIXED_FREQ_CENTER_HZ = [167.0]
FIXED_TOTAL_PATCHES = 1000
FIXED_PATCH_CYCLES = 4


def reset_config():
    CONFIG.clear()
    CONFIG.update(copy.deepcopy(BASE_CONFIG))


def enforce_fixed_patch_cycles(data_cfg):
    data_cfg["num_total_patches"] = FIXED_TOTAL_PATCHES
    data_cfg["patch_num_cycles"] = FIXED_PATCH_CYCLES
    data_cfg["patch_hop_cycles"] = FIXED_PATCH_CYCLES
    data_cfg.pop("patch_num_samples", None)
    data_cfg.pop("patch_hop_samples", None)


def assert_fixed_patch_geometry(data_cfg):
    if int(data_cfg.get("num_total_patches", -1)) != FIXED_TOTAL_PATCHES:
        raise AssertionError(
            f"num_total_patches must be {FIXED_TOTAL_PATCHES}, "
            f"got {data_cfg.get('num_total_patches')}"
        )
    if int(data_cfg.get("patch_num_cycles", -1)) != FIXED_PATCH_CYCLES:
        raise AssertionError(
            f"patch_num_cycles must be {FIXED_PATCH_CYCLES}, "
            f"got {data_cfg.get('patch_num_cycles')}"
        )
    if int(data_cfg.get("patch_hop_cycles", -1)) != FIXED_PATCH_CYCLES:
        raise AssertionError(
            f"patch_hop_cycles must be {FIXED_PATCH_CYCLES}, "
            f"got {data_cfg.get('patch_hop_cycles')}"
        )
    if data_cfg.get("patch_num_samples") is not None:
        raise AssertionError(
            "patch_num_samples must be unset/None so patch length is "
            "patch_num_cycles * num_probes"
        )
    if data_cfg.get("patch_hop_samples") is not None:
        raise AssertionError(
            "patch_hop_samples must be unset/None so hop length is "
            "patch_hop_cycles * num_probes"
        )


def parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def combo_name(probe_count, half_band_hz, init_mode):
    band = str(float(half_band_hz)).replace(".", "p")
    return f"init_{init_mode}__probes_{int(probe_count)}__band_{band}hz"


def parse_str_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def validate_init_mode(init_mode):
    mode = str(init_mode).strip().lower()
    if mode not in ("center", "oracle", "data_driven"):
        raise ValueError("init_mode must be one of center, oracle, data_driven")
    return mode


def first_n_values(values, n, name):
    if len(values) < n:
        raise ValueError(f"{name} must contain at least {n} values")
    return [float(value) for value in values[:n]]


def uniform_probe_angles_degrees(probe_count):
    probe_count = int(probe_count)
    if probe_count <= 0:
        raise ValueError("probe_count must be positive")
    step = 360.0 / float(probe_count)
    return [float(i * step) for i in range(probe_count)]


def frequency_support_from_config():
    freq_cfg = CONFIG["frequency"]
    absolute_half_band = freq_cfg.get("absolute_half_band_hz")
    if absolute_half_band is not None:
        return compute_frequency_support(
            freq_center_hz=freq_cfg["center_hz"],
            absolute_half_band_hz=absolute_half_band,
        )
    return compute_frequency_support(
        freq_center_hz=freq_cfg["center_hz"],
        relative_half_band=freq_cfg["relative_half_band"],
    )


def true_frequency_at_position(freq_lower, freq_upper, position):
    if position < 0.0 or position > 1.0:
        raise ValueError("true frequency position must be in [0, 1]")
    return [
        float(lower + position * (upper - lower))
        for lower, upper in zip(freq_lower, freq_upper)
    ]


def configure_run(
    run_dir,
    seed,
    probe_count,
    half_band_hz,
    init_mode,
    true_frequency_position,
):
    init_mode = validate_init_mode(init_mode)
    CONFIG["seed"] = int(seed)
    CONFIG["run_dir"] = run_dir
    CONFIG["sweep_name"] = "direct_global_init_bandwidth"
    CONFIG["sweep_value"] = {
        "init_mode": init_mode,
        "probe_count": int(probe_count),
        "absolute_half_band_hz": float(half_band_hz),
    }
    CONFIG["model"]["inference_parameterization"] = "direct_global"

    data_cfg = CONFIG["data"]
    data_cfg["num_harmonics"] = FIXED_NUM_HARMONICS
    data_cfg["num_probes"] = int(probe_count)
    data_cfg["probes"] = uniform_probe_angles_degrees(probe_count)
    enforce_fixed_patch_cycles(data_cfg)
    assert_fixed_patch_geometry(data_cfg)

    freq_cfg = CONFIG["frequency"]
    freq_cfg["center_hz"] = list(FIXED_FREQ_CENTER_HZ)
    freq_cfg["absolute_half_band_hz"] = float(half_band_hz)
    freq_cfg.setdefault("posterior_init", {})
    freq_cfg["posterior_init"]["mode"] = init_mode

    signal_cfg = CONFIG["signal"]
    signal_cfg["single_instance_parameter_source"] = "fixed"
    signal_cfg["true_amp_real"] = first_n_values(
        signal_cfg.get("true_amp_real", []),
        FIXED_NUM_HARMONICS,
        "signal.true_amp_real",
    )
    signal_cfg["true_amp_imag"] = first_n_values(
        signal_cfg.get("true_amp_imag", []),
        FIXED_NUM_HARMONICS,
        "signal.true_amp_imag",
    )
    signal_cfg["amp_real_center_m"] = first_n_values(
        signal_cfg.get("amp_real_center_m", []),
        FIXED_NUM_HARMONICS,
        "signal.amp_real_center_m",
    )
    signal_cfg["amp_imag_center_m"] = first_n_values(
        signal_cfg.get("amp_imag_center_m", []),
        FIXED_NUM_HARMONICS,
        "signal.amp_imag_center_m",
    )

    freq_lower, freq_upper, _freq_center, _freq_half = frequency_support_from_config()
    true_freq_hz = true_frequency_at_position(
        freq_lower=freq_lower,
        freq_upper=freq_upper,
        position=float(true_frequency_position),
    )
    signal_cfg["true_frequency_hz"] = true_freq_hz

    CONFIG["checkpoint"]["dir"] = os.path.join(run_dir, "checkpoints")
    CONFIG["checkpoint"]["name"] = "latest.pt"
    CONFIG["checkpoint"]["resume_from"] = None
    CONFIG["logging"]["tensorboard_dir"] = os.path.join(run_dir, "tensorboard")
    CONFIG["logging"]["curve_dir"] = os.path.join(run_dir, "curves")
    return true_freq_hz, freq_lower, freq_upper


def read_metrics(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Expected metrics.json not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("last_metrics", payload)


def write_rows_csv(rows, csv_path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_condition_diagnostics(freq_lower, freq_upper):
    assert_fixed_patch_geometry(CONFIG["data"])
    dataset = train_main._build_dataset(
        num_sequences=1,
        seed=CONFIG.get("seed", 42),
        data_cfg=CONFIG["data"],
        signal_cfg=CONFIG["signal"],
        freq_lower=freq_lower,
        freq_upper=freq_upper,
    )
    sample = dataset[0]
    train = sample["train"]
    t_local = _to_numpy(train["t_local"]).astype(np.float64)
    t0_abs = _to_numpy(train["patch_t0_abs"]).astype(np.float64)
    true_freq = _to_numpy(sample["true_freq_hz"]).astype(np.float64)
    true_amp = (
        _to_numpy(sample["true_amp_real"]).astype(np.float64)
        + 1j * _to_numpy(sample["true_amp_imag"]).astype(np.float64)
    )

    dictionary_cond = []
    jacobian_cond = []
    for p in range(t_local.shape[0]):
        phi = np.exp(1j * 2.0 * np.pi * t_local[p, :, None] * true_freq[None, :])
        dictionary_cond.append(float(np.linalg.cond(phi)))
        c_local = true_amp * np.exp(1j * 2.0 * np.pi * true_freq * t0_abs[p])
        jac = 1j * 2.0 * np.pi * t_local[p, :, None] * c_local[None, :] * phi
        jacobian_cond.append(float(np.linalg.cond(jac)))

    return {
        "dictionary_cond_mean": float(np.mean(dictionary_cond)),
        "dictionary_cond_p95": float(np.percentile(dictionary_cond, 95)),
        "jacobian_cond_mean": float(np.mean(jacobian_cond)),
        "jacobian_cond_p95": float(np.percentile(jacobian_cond, 95)),
    }


def profile_equivalent_valley_summary(
    metrics,
    true_freq_hz,
    freq_lower,
    freq_upper,
    harmonic_indices,
    grid_points,
    equiv_delta_nll,
    min_separation_hz,
):
    if not harmonic_indices:
        return {
            "profile_equiv_valley_any": "",
            "profile_equiv_valley_count": "",
            "profile_equiv_valley_min_delta_nll": "",
            "profile_equiv_valley_harmonics": "",
        }

    estimated_freq_hz = estimated_frequency_from_metrics(
        metrics=metrics,
        num_harmonics=len(true_freq_hz),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan = build_scan_tensors(
        freq_lower=freq_lower,
        freq_upper=freq_upper,
        device=device,
    )

    equivalent = []
    min_delta = None
    for harmonic_index in harmonic_indices:
        rows = scan_profile_likelihood(
            scan=scan,
            true_freq_hz=true_freq_hz,
            estimated_freq_hz=estimated_freq_hz,
            freq_lower=freq_lower,
            freq_upper=freq_upper,
            harmonic_index=harmonic_index,
            grid_points=grid_points,
        )
        for fixed_mode in ("others_true", "others_estimated"):
            minima = find_local_minima(rows, fixed_mode)
            if len(minima) < 2:
                continue
            best = minima[0]
            for candidate in minima[1:]:
                separation = abs(
                    candidate["scan_frequency_hz"] - best["scan_frequency_hz"]
                )
                delta = candidate["profile_nll_full"] - best["profile_nll_full"]
                if separation >= min_separation_hz and delta <= equiv_delta_nll:
                    equivalent.append(
                        {
                            "harmonic_index": int(harmonic_index),
                            "fixed_mode": fixed_mode,
                            "frequency_hz": candidate["scan_frequency_hz"],
                            "delta_nll": delta,
                            "separation_hz": separation,
                        }
                    )
                    min_delta = delta if min_delta is None else min(min_delta, delta)
                    break

    return {
        "profile_equiv_valley_any": int(bool(equivalent)),
        "profile_equiv_valley_count": len(equivalent),
        "profile_equiv_valley_min_delta_nll": "" if min_delta is None else float(min_delta),
        "profile_equiv_valley_harmonics": " ".join(
            str(item["harmonic_index"]) for item in equivalent
        ),
        "profile_equiv_valley_details": equivalent,
    }


def summarize_run(
    init_mode,
    probe_count,
    half_band_hz,
    seed,
    run_dir,
    true_frequency_position,
    true_freq_hz,
    metrics,
    condition_diag,
    profile_diag,
):
    joint_success = metrics.get("joint_amp_freq_success_rate_mean", "")
    freq_success = metrics.get("freq_success_rate_mean", "")
    row = {
        "init_mode": validate_init_mode(init_mode),
        "probe_count": int(probe_count),
        "absolute_half_band_hz": float(half_band_hz),
        "seed": int(seed),
        "run_dir": run_dir,
        "true_frequency_position": float(true_frequency_position),
        "patch_num_cycles": FIXED_PATCH_CYCLES,
        "patch_hop_cycles": FIXED_PATCH_CYCLES,
        "num_total_patches": FIXED_TOTAL_PATCHES,
        "patch_num_samples_override": CONFIG["data"].get("patch_num_samples", ""),
        "patch_hop_samples_override": CONFIG["data"].get("patch_hop_samples", ""),
        "patch_samples_per_patch": int(FIXED_PATCH_CYCLES * int(probe_count)),
        "probe_angles_deg": " ".join(
            f"{angle:.6g}" for angle in CONFIG["data"]["probes"]
        ),
        "freq_rmse_hz_mean": metrics.get("freq_rmse_hz_mean", ""),
        "freq_nrmse_band_mean": metrics.get("freq_nrmse_band_mean", ""),
        "freq_success_rate_mean": freq_success,
        "joint_amp_freq_success_rate_mean": joint_success,
        "freq_failure": "" if freq_success == "" else int(float(freq_success) < 1.0),
        "joint_failure": "" if joint_success == "" else int(float(joint_success) < 1.0),
        "posterior_std_hz_mean": metrics.get("posterior_std_hz_mean", ""),
        "global_freq_coverage_68": metrics.get("global_freq_coverage_68", ""),
        "global_freq_coverage_95": metrics.get("global_freq_coverage_95", ""),
        "ls_cond_p95": metrics.get("ls_cond_p95", ""),
        "fusion_effective_num_patches_mean": metrics.get(
            "fusion_effective_num_patches_mean",
            "",
        ),
    }
    row.update(condition_diag)
    row.update(profile_diag)
    for idx, value in enumerate(true_freq_hz, start=1):
        row[f"true_freq_h{idx}_hz"] = float(value)
        row[f"freq_rmse_h{idx}_hz"] = metrics.get(f"freq_rmse_h{idx}_hz", "")
    return row


def _numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        values.append(float(value))
    return values


def _mean(values):
    return "" if not values else sum(values) / float(len(values))


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["init_mode"], row["probe_count"], row["absolute_half_band_hz"])
        grouped.setdefault(key, []).append(row)

    aggregate = []
    for (init_mode, probe_count, half_band_hz), group in sorted(grouped.items()):
        joint_failures = _numeric_values(group, "joint_failure")
        freq_failures = _numeric_values(group, "freq_failure")
        rmse_all = _numeric_values(group, "freq_rmse_hz_mean")
        success_rows = [
            row
            for row in group
            if row.get("joint_failure", "") != "" and int(row["joint_failure"]) == 0
        ]
        rmse_success = _numeric_values(success_rows, "freq_rmse_hz_mean")
        row = {
            "init_mode": init_mode,
            "probe_count": int(probe_count),
            "absolute_half_band_hz": float(half_band_hz),
            "num_seeds": len(group),
            "joint_failure_rate": _mean(joint_failures),
            "freq_failure_rate": _mean(freq_failures),
            "freq_rmse_hz_mean_all": _mean(rmse_all),
            "conditional_freq_rmse_hz_mean_success": _mean(rmse_success),
            "posterior_coverage_68_mean": _mean(
                _numeric_values(group, "global_freq_coverage_68")
            ),
            "posterior_coverage_95_mean": _mean(
                _numeric_values(group, "global_freq_coverage_95")
            ),
            "ls_cond_p95_mean": _mean(_numeric_values(group, "ls_cond_p95")),
            "dictionary_cond_p95_mean": _mean(
                _numeric_values(group, "dictionary_cond_p95")
            ),
            "jacobian_cond_p95_mean": _mean(
                _numeric_values(group, "jacobian_cond_p95")
            ),
            "profile_equiv_valley_rate": _mean(
                _numeric_values(group, "profile_equiv_valley_any")
            ),
            "profile_equiv_valley_count_mean": _mean(
                _numeric_values(group, "profile_equiv_valley_count")
            ),
        }
        aggregate.append(row)
    return aggregate


def write_summary(rows, root):
    os.makedirs(root, exist_ok=True)
    raw_json = os.path.join(root, "probe_bandwidth_joint_sweep_rows.json")
    raw_csv = os.path.join(root, "probe_bandwidth_joint_sweep_rows.csv")
    aggregate_json = os.path.join(root, "probe_bandwidth_joint_sweep_aggregate.json")
    aggregate_csv = os.path.join(root, "probe_bandwidth_joint_sweep_aggregate.csv")

    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_rows_csv(rows, raw_csv)

    aggregate = aggregate_rows(rows)
    with open(aggregate_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    write_rows_csv(aggregate, aggregate_csv)

    print(f"Saved rows JSON: {raw_json}")
    print(f"Saved rows CSV: {raw_csv}")
    print(f"Saved aggregate JSON: {aggregate_json}")
    print(f"Saved aggregate CSV: {aggregate_csv}")


def resolve_harmonic_indices(text, num_harmonics):
    if str(text).strip().lower() in ("all", "*"):
        return list(range(1, int(num_harmonics) + 1))
    values = parse_int_list(text)
    for value in values:
        if value < 1 or value > int(num_harmonics):
            raise ValueError("profile harmonic indices must be in [1, num_harmonics]")
    return values


def run(args):
    rows = []
    num_harmonics = FIXED_NUM_HARMONICS
    harmonic_indices = (
        []
        if args.skip_profile_scan
        else resolve_harmonic_indices(args.profile_harmonics, num_harmonics)
    )
    for init_mode in parse_str_list(args.init_modes):
        init_mode = validate_init_mode(init_mode)
        for probe_count in parse_int_list(args.probe_counts):
            for half_band_hz in parse_float_list(args.half_bands_hz):
                for seed in parse_int_list(args.seeds):
                    reset_config()
                    run_dir = os.path.join(
                        args.root,
                        combo_name(probe_count, half_band_hz, init_mode),
                        f"seed_{int(seed)}",
                    )
                    os.makedirs(run_dir, exist_ok=True)
                    true_freq_hz, freq_lower, freq_upper = configure_run(
                        run_dir=run_dir,
                        seed=seed,
                        probe_count=probe_count,
                        half_band_hz=half_band_hz,
                        init_mode=init_mode,
                        true_frequency_position=args.true_frequency_position,
                    )
                    with open(
                        os.path.join(run_dir, "config.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(CONFIG, f, indent=2)

                    if not args.skip_train:
                        assert_fixed_patch_geometry(CONFIG["data"])
                        print("=" * 80)
                        print(
                            "Running joint probe/bandwidth sweep, "
                            f"init={init_mode}, "
                            f"M={probe_count}, B={half_band_hz:g} Hz, "
                            f"K={CONFIG['data']['num_harmonics']}, "
                            f"centers={CONFIG['frequency']['center_hz']}, "
                            f"patch_cycles={CONFIG['data']['patch_num_cycles']}, "
                            f"total_patches={CONFIG['data']['num_total_patches']}, "
                            f"patch_samples={FIXED_PATCH_CYCLES * int(probe_count)}, "
                            f"r={args.true_frequency_position:g}, "
                            f"seed={seed}, run_dir={run_dir}"
                        )
                        print("=" * 80)
                        train_main.main()

                    metrics = read_metrics(run_dir)
                    condition_diag = compute_condition_diagnostics(
                        freq_lower=freq_lower,
                        freq_upper=freq_upper,
                    )
                    profile_diag = profile_equivalent_valley_summary(
                        metrics=metrics,
                        true_freq_hz=true_freq_hz,
                        freq_lower=freq_lower,
                        freq_upper=freq_upper,
                        harmonic_indices=harmonic_indices,
                        grid_points=args.profile_grid_points,
                        equiv_delta_nll=args.equiv_delta_nll,
                        min_separation_hz=args.equiv_min_separation_hz,
                    )
                    row = summarize_run(
                        init_mode=init_mode,
                        probe_count=probe_count,
                        half_band_hz=half_band_hz,
                        seed=seed,
                        run_dir=run_dir,
                        true_frequency_position=args.true_frequency_position,
                        true_freq_hz=true_freq_hz,
                        metrics=metrics,
                        condition_diag=condition_diag,
                        profile_diag=profile_diag,
                    )
                    rows.append(row)
                    write_summary(rows, args.root)
    write_summary(rows, args.root)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Direct-global initialization/bandwidth sweep for a single "
            "frequency component centered at 167 Hz. Fixed M=6, "
            "num_total_patches=1000, patch_num_cycles=4, and "
            "true-frequency r=0.1."
        )
    )
    parser.add_argument(
        "--probe-counts",
        default=",".join(str(x) for x in DEFAULT_PROBE_COUNTS),
        help="Comma-separated M values. Default: 6",
    )
    parser.add_argument(
        "--init-modes",
        default=",".join(DEFAULT_INIT_MODES),
        help="Comma-separated init modes: center, oracle, data_driven. Default: center,oracle",
    )
    parser.add_argument(
        "--half-bands-hz",
        default=",".join(str(x) for x in DEFAULT_HALF_BANDS_HZ),
        help="Comma-separated absolute half-band B values in Hz. Default: 15,...,120",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated seeds. Default: 0,...,9",
    )
    parser.add_argument(
        "--true-frequency-position",
        type=float,
        default=DEFAULT_TRUE_FREQUENCY_POSITION,
        help="True-frequency position r inside each search band. Default: 0.1",
    )
    parser.add_argument(
        "--root",
        default="artifacts/profile_LS_ELBO/probe_bandwidth_joint_sweep",
        help="Root directory for sweep artifacts.",
    )
    parser.add_argument(
        "--profile-harmonics",
        default="all",
        help="Profile-likelihood harmonics to scan, e.g. all or 4. Default: all",
    )
    parser.add_argument(
        "--profile-grid-points",
        type=int,
        default=101,
        help="Grid points per one-dimensional profile scan. Default: 101",
    )
    parser.add_argument(
        "--equiv-delta-nll",
        type=float,
        default=2.0,
        help="A second local minimum within this delta NLL is an equivalent valley.",
    )
    parser.add_argument(
        "--equiv-min-separation-hz",
        type=float,
        default=1.0,
        help="Minimum separation from best local minimum for an equivalent valley.",
    )
    parser.add_argument(
        "--skip-profile-scan",
        action="store_true",
        help="Skip profile-likelihood valley detection for a faster sweep.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing run_dir/metrics.json instead of training first.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
