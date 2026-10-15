"""Closed-form coordinate implementation of stage2_derivation.tex.

This is the TeX version, not the MC collapsed bound in elbo_marginal.py:
  * q(S) is represented exactly by enumerating all K-sparse subsets.
  * q(c_S | S) is updated in closed form for every subset.
  * q(S), tau_k^2, and sigma_w^2 use the stationary closed-form updates.
  * q(f_k)=N(mu_k,sigma_k^2) is updated one coordinate at a time by maximizing
    the analytic two-scalar objective g_k from the TeX residual formula.

The implementation is single-instance NumPy/SciPy at the core because the TeX
algorithm is inherently subset-enumerating and coordinate based.  A small batch
wrapper accepts either NumPy arrays or Torch tensors and returns tensors when
the input was Torch.
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

TWO_PI = 2.0 * math.pi

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# Each instance is a small dense solve, so multi-threaded BLAS buys nothing and
# oversubscribes the box once instances run in parallel.  Pinning must happen at
# import time (before numpy/scipy bind their thread pools) or the pool ends up
# SLOWER than serial -- measured 67.8s vs 50.0s at B=100 when left unpinned.
for _v in _BLAS_VARS:
    os.environ.setdefault(_v, "1")

# Measured on a 96-core Xeon 8260, B=100, n_sweeps=40 (BLAS pinned):
#   n_jobs   1 -> 33.9s | 4 -> 9.1s | 8 -> 5.2s | 16 -> 4.3s | 32 -> 5.3s
# Past ~16 the per-process spawn + pickling overhead outweighs the gain.
_DEFAULT_JOBS = min(16, os.cpu_count() or 1)


def _init_worker():
    """Pin BLAS to a single thread inside each pool worker."""
    for v in _BLAS_VARS:
        os.environ[v] = "1"


def _as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_tensor(x, like):
    if not hasattr(like, "detach"):
        return x
    import torch
    if np.issubdtype(np.asarray(x).dtype, np.integer):
        return torch.as_tensor(x, device=like.device, dtype=torch.long)
    if np.iscomplexobj(x):
        return torch.as_tensor(x, device=like.device, dtype=torch.complex64)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _mean_atoms(mu, sigma2, t):
    return (np.exp(1j * TWO_PI * np.outer(t, mu))
            * np.exp(-2.0 * math.pi**2 * np.outer(t**2, sigma2)))


def _expected_stats(mu, sigma2, t, y):
    Abar = _mean_atoms(mu, sigma2, t)
    J = Abar.conj().T @ Abar
    np.fill_diagonal(J, float(len(t)))
    h = Abar.conj().T @ y
    return Abar, J, h


def _refresh_column(Abar, J, h, k, mu, sigma2, t, y):
    """Recompute only what changed after mu[k], sigma2[k] moved.

    _update_frequency touches a single coordinate, so Abar changes in column k
    only, which makes J change in row/column k only and h in entry k only.
    Rebuilding them wholesale costs O(N*Kc^2) per coordinate; this is O(N*Kc).
    Abar/J/h are modified in place and also returned for call-site symmetry.
    """
    ak = (np.exp(1j * TWO_PI * t * mu[k])
          * np.exp(-2.0 * math.pi**2 * t**2 * sigma2[k]))
    Abar[:, k] = ak
    col = Abar.conj().T @ ak            # <a_j, a_k> for every j
    J[:, k] = col
    J[k, :] = col.conj()
    J[k, k] = float(len(t))             # diagonal is pinned, as in _expected_stats
    h[k] = np.vdot(ak, y)
    return Abar, J, h


def _amplitude_posterior(J, h, tau2, sw2, S):
    idx = np.asarray(S, dtype=int)
    A = J[np.ix_(idx, idx)] / sw2 + np.diag(1.0 / tau2[idx])
    C = np.linalg.inv(A)
    c = C @ h[idx] / sw2
    return A, c, C


def _subset_score(J, h, tau2, sw2, rho, S):
    idx = np.asarray(S, dtype=int)
    A = J[np.ix_(idx, idx)] / sw2 + np.diag(1.0 / tau2[idx])
    L = np.linalg.cholesky(A)
    z = np.linalg.solve(L, h[idx])
    logit = np.log(rho[idx] / (1.0 - rho[idx]))
    return (np.real(z.conj() @ z) / sw2**2
            - 2.0 * np.log(np.real(np.diag(L))).sum()
            - np.log(tau2[idx]).sum()
            + logit.sum())


def _subset_posterior(J, h, tau2, sw2, rho, subsets):
    F = np.array([_subset_score(J, h, tau2, sw2, rho, S) for S in subsets])
    m = F.max()
    w = np.exp(F - m)
    qS = w / w.sum()
    return F, qS, m + np.log(w.sum())


def _residuals(Abar, y, subsets, qS, chats, covs, Kc, sub_idx=None):
    """Eq. residuals with the self term excluded.

    Vectorised over the B = C(Kc,K) subsets.  The scalar loop this replaces was
    the single largest cost in the profile, and it is pure Python overhead: the
    subset list is fixed across sweeps, so the whole accumulation is a scatter-add
    of B small dense blocks.  sub_idx is the [B,K] integer view of `subsets`,
    passed in by the caller so it is built once per instance rather than per call.
    """
    idx = np.asarray(subsets, dtype=int) if sub_idx is None else sub_idx
    w = np.asarray(qS, dtype=float)                     # [B]
    c = np.asarray(chats)                               # [B,K]
    C = np.asarray(covs)                                # [B,K,K]

    # alpha[kappa] = sum_{S ni kappa} q(S) conj(c_kappa^S)
    alpha = np.zeros(Kc, dtype=np.complex128)
    np.add.at(alpha, idx.ravel(), (w[:, None] * np.conj(c)).ravel())

    # V[ell,kappa] = sum_S q(S) (c_ell^S conj(c_kappa^S) + C_{ell,kappa}^S),
    # with the ell == kappa self term excluded.
    blk = c[:, :, None] * np.conj(c)[:, None, :] + C    # [B,K,K], (ell, kappa)
    K = idx.shape[1]
    eye = np.eye(K, dtype=bool)
    blk[:, eye] = 0.0                                   # drop the self term
    blk *= w[:, None, None]
    rows = np.broadcast_to(idx[:, :, None], blk.shape)
    cols = np.broadcast_to(idx[:, None, :], blk.shape)
    V = np.zeros((Kc, Kc), dtype=np.complex128)
    np.add.at(V, (rows.ravel(), cols.ravel()), blk.ravel())

    return y[:, None] * alpha[None, :] - Abar @ V


def _g_value(mu_k, sigma2_k, r, t, sw2):
    damp = np.exp(-2.0 * math.pi**2 * sigma2_k * t**2)
    return (2.0 * np.sum(damp * np.real(r * np.exp(-1j * TWO_PI * mu_k * t))) / sw2
            + 0.5 * np.log(sigma2_k))


def _g_grad(mu_k, sigma2_k, r, t, sw2):
    beta = 2.0 * np.abs(r) * np.exp(-2.0 * math.pi**2 * sigma2_k * t**2) / sw2
    theta = np.angle(r) - TWO_PI * mu_k * t
    c = np.cos(theta)
    s = np.sin(theta)
    d_mu = np.sum(beta * TWO_PI * t * s)
    d_s2 = -np.sum(beta * 2.0 * math.pi**2 * t**2 * c) + 0.5 / sigma2_k
    return np.array([d_mu, d_s2])


def _update_frequency_point(mu_k, r, t, sw2, band=None):
    """点估计版频率更新: sigma^2 恒 0, 只对 mu 做一维优化。

    目标 = _g_value 去掉熵项后的相干项 (sigma^2=0 ⇒ damp≡1):
        g(mu) = 2/sw2 * sum Re( r * exp(-i*2pi*mu*t) )
    ⚠ 这里**没有** 0.5*log(sigma^2) —— 点估计的目标函数本就不含微分熵,
      而不是把 -inf 硬塞进去 (见 FREQ_SIGMA0 的说明)。
    """
    def neg(x):
        mu = x[0]
        th = TWO_PI * mu * t
        val = 2.0 * np.sum(np.real(r * np.exp(-1j * th))) / sw2
        # d/dmu: r*exp(-i*2pi*mu*t) 的实部对 mu 求导 = 2*|r|*sin(angle(r)-2pi*mu*t)*2pi*t
        beta = 2.0 * np.abs(r) / sw2
        theta = np.angle(r) - th
        d_mu = np.sum(beta * TWO_PI * t * np.sin(theta))
        return -val, -np.array([d_mu])

    lo_mu, hi_mu = (None, None) if band is None else band
    old = -neg(np.array([mu_k]))[0]
    res = minimize(neg, np.array([float(mu_k)]), jac=True,
                   method="L-BFGS-B", bounds=[(lo_mu, hi_mu)])
    if res.success and -res.fun >= old:
        return float(res.x[0]), 0.0
    return float(mu_k), 0.0


def _update_frequency(mu_k, sigma2_k, r, t, sw2, band=None):
    """Maximize g_k over (mu, log sigma^2) with analytic gradient.

    The upper bound on sigma^2 is tied to the box half-width, not to a fixed
    number: mu_k is already confined to +-hw, so a posterior wider than hw carries
    no localisation information -- the distribution would cover the whole feasible
    region.  Left unbounded, sigma^2 runs away and the coherence weight
    exp(-2 pi^2 sigma^2 t^2) UNDERFLOWS to zero, which flattens g_k in mu and lets
    mu random-walk (measured at 0dB/100seq: ~9% of components ended past hw^2).
    hw^2 sits at the healthy/diverged split there (8.9% above it, median sigma^2
    being 1.8e-6).

    Underflow starts at exp(-745), i.e. sigma^2 > 745 / (2 pi^2 t_max^2) -- the
    2 pi^2 ~ 19.74 factor matters: with t_max = 3.87 s the cap is 2.52, not the
    47.3 one gets from 745/t_max^2 alone.  So hw^2 must satisfy

        (BAND_FRAC * grid_step)^2 < 745 / (2 pi^2 t_max^2)

    which at t_max = 3.87 s allows hw < 1.59 Hz; with BAND_FRAC = 0.5 that caps
    grid_step at ~3.2 Hz.  A 2Hz grid (hw = 1) clears it by only 2.5x, a 1Hz grid
    (hw = 0.5) by 10x -- so the bound below takes the min of the two rather than
    trusting hw alone.  (Note this is underflow of the damping term; what would
    actually overflow is exp(x[1]) = sigma^2 itself inside neg().)

    Falls back to the old fixed decade bound when there is no box (hw unknown).
    """
    lo_mu, hi_mu = (None, None) if band is None else band
    # 阻尼项 exp(-2 pi^2 s2 t^2) 下溢的硬上限 (exp(-745) 处), 与 hw 取较小者。
    tm = float(np.max(np.abs(t)))
    s2_underflow = 745.0 / (2.0 * math.pi ** 2 * max(tm, 1e-12) ** 2)
    hi_s2 = None
    if lo_mu is not None and hi_mu is not None:
        hw = 0.5 * (float(hi_mu) - float(lo_mu))
        if hw > 0:
            hi_s2 = np.log(min(hw ** 2, s2_underflow))
    # 两侧都必须有界, 方向不同但同样致命:
    #   无上界 -> sigma^2 跑大, exp(-2 pi^2 sigma^2 t^2) 溢出 (实测 301 次 overflow)
    #   无下界 -> sigma^2 跑到 0, log(sigma2_k) 与 0.5/sigma2_k 除零 -> NaN
    # 下界 1e-12 对应 sigma=1e-6 Hz, 远低于任何有意义的频率分辨率 (CRB 量级 1e-4 Hz,
    # 实测健康分量 sigma 中位 1.4e-3 Hz), 故不会误伤真实解。
    lo_s2 = np.log(1e-12)
    if hi_s2 is None:                   # 无箱约束: 仍用下溢上限, 不再用 1e6
        hi_s2 = np.log(min(1e6, s2_underflow))
    bounds = [(lo_mu, hi_mu), (lo_s2, hi_s2)]

    def neg(x):
        s2 = np.exp(x[1])
        val = _g_value(x[0], s2, r, t, sw2)
        grad = _g_grad(x[0], s2, r, t, sw2)
        return -val, -np.array([grad[0], grad[1] * s2])

    old = _g_value(mu_k, sigma2_k, r, t, sw2)
    res = minimize(neg, np.array([mu_k, np.log(sigma2_k)]), jac=True,
                   method="L-BFGS-B", bounds=bounds)
    if res.success and -res.fun >= old:
        return float(res.x[0]), float(np.exp(res.x[1]))
    return float(mu_k), float(sigma2_k)


def _model_avg_amplitude(subsets, qS, chats, covs, Kc, eps,
                         S_star=None, c_star=None, C_star=None):
    """Amplitude posterior mean/variance, model-averaged over q(S).

    stage2_derivation.tex, Outputs:
        c_k = pi_k^-1 * sum_{S ni k} q(S) c_k^(S)
    即在**全部**含 k 的子集上做模型平均, 而不是只取 MAP 子集 S*。
    精确枚举 q(S) 的全部意义就在这里; 只取 S* 等于把它扔掉, q(S) 越分散丢得越多
    (q(S) 退化成 one-hot 时两者恒等 —— 高 SNR 实测 q_entropy -> 0, 故只在低 SNR
     有差别)。方差用混合分布的全方差公式: E[Var] + Var[E], 后者是跨子集的离散度。

    AMP_MODEL_AVG=0 时退回只取 S* (此时 S_star/c_star/C_star 必须给)。
    抽成函数是为了 trace= 能在**每个 sweep** 上复用同一段口径, 见 refine 的 trace 钩子。
    """
    c_map = np.zeros(Kc, dtype=np.complex128)
    v_map = np.zeros(Kc, dtype=float)
    if not AMP_MODEL_AVG:
        c_map[S_star] = c_star
        v_map[S_star] = np.real(np.diag(C_star))
        return c_map, v_map
    for kk in range(Kc):
        wsum = 0.0
        m1 = 0.0 + 0.0j
        m2 = 0.0                     # sum_S q(S) (C_kk^(S) + |c_k^(S)|^2)
        for S, w, c, C in zip(subsets, qS, chats, covs):
            if kk not in S:
                continue
            j = S.index(kk)
            wsum += w
            m1 += w * c[j]
            m2 += w * (np.real(C[j, j]) + abs(c[j]) ** 2)
        if wsum > eps:
            c_map[kk] = m1 / wsum
            v_map[kk] = max(m2 / wsum - abs(c_map[kk]) ** 2, 0.0)
    return c_map, v_map


def refine_stage2_tex_instance(y, t, rho, mu0, sigma0, K, n_sweeps=40,
                               tol=1e-8, bands=None, verbose=False,
                               eps=1e-12, jacobi=None, trace=False):
    """Run Algorithm 1 from stage2_derivation.tex for one instance.

    jacobi: None -> module default JACOBI (env TEX_JACOBI).  See the comment on
    the sweep loop for what the two orderings mean.

    trace=True additionally returns per-sweep "mu_hist" [n,Kc] and "amp_hist"
    [n,Kc] alongside "elbo" -- the same model-averaged amplitude the final output
    uses, evaluated at every sweep.  Off by default: it costs one extra
    _model_avg_amplitude per sweep and is only wanted by the convergence plot.
    ⚠ the histories stop where the tol break stops, so they can be shorter than
      n_sweeps; the caller pads (see eval/trace_convergence.py).
    """
    if jacobi is None:
        jacobi = JACOBI
    y = np.asarray(y, dtype=np.complex128)
    t = np.asarray(t, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if RHO_UNIFORM:                     # 均匀先验: logit=0, 子集只看数据证据
        rho = np.full_like(rho, 0.5)
    if not NOCLIP:
        rho = rho.clip(1e-6, 1.0 - 1e-6)
    mu = np.asarray(mu0, dtype=float).copy()
    sigma2 = np.asarray(sigma0, dtype=float) ** 2
    if not NOCLIP:
        sigma2 = np.maximum(sigma2, eps)
    if FREQ_SIGMA0:                     # 点估计: 频率后验退化成 delta (见 FREQ_SIGMA0)
        sigma2 = np.zeros_like(sigma2)
    Kc = len(mu)
    N = len(t)
    subsets = [list(S) for S in combinations(range(Kc), K)]
    sub_idx = np.asarray(subsets, dtype=int)        # [B,K] view, reused every sweep

    Phi0 = np.exp(1j * TWO_PI * np.outer(t, mu))
    c0 = np.linalg.lstsq(Phi0, y, rcond=None)[0]
    tau2 = np.abs(c0) ** 2
    sw2 = float(np.mean(np.abs(y - Phi0 @ c0) ** 2))
    if not NOCLIP:
        tau2 = np.maximum(tau2, eps)
        sw2 = max(sw2, eps)

    hist = []
    mu_hist, amp_hist = [], []
    for sweep in range(n_sweeps):
        # Two orderings of the frequency step, selected by `jacobi`:
        #
        #   jacobi=True  (Algorithm 1 of stage2_derivation.tex): build J/h, the
        #     B subset posteriors and the residual matrix R ONCE, then update all
        #     Kc frequencies from that same R.  Line 13 of the algorithm forms
        #     every r_kappa in one N x Kc product, which is the whole point of
        #     accumulating V first.
        #
        #   jacobi=False (Gauss-Seidel): refresh the closed-form factors after
        #     every single coordinate.  Strictly more up-to-date information per
        #     step, at Kc times the cost per sweep.
        #
        # The TeX describes the first; the implementation historically did the
        # second.  Keep both so the difference can be measured rather than assumed.
        Abar, J, h = _expected_stats(mu, sigma2, t, y)
        if jacobi:
            chats, covs = [], []
            for S in subsets:
                _, c, C = _amplitude_posterior(J, h, tau2, sw2, S)
                chats.append(c)
                covs.append(C)
            F, qS, _ = _subset_posterior(J, h, tau2, sw2, rho, subsets)
            R = _residuals(Abar, y, subsets, qS, chats, covs, Kc, sub_idx)
            for k in range(Kc):
                band = None if bands is None else bands[k]
                if FREQ_SIGMA0:
                    mu[k], sigma2[k] = _update_frequency_point(
                        mu[k], R[:, k], t, sw2, band)
                else:
                    mu[k], sigma2[k] = _update_frequency(mu[k], sigma2[k], R[:, k],
                                                         t, sw2, band)
        else:
            for k in range(Kc):
                if k:                   # k=0 uses the stats just built above
                    Abar, J, h = _refresh_column(Abar, J, h, k - 1, mu, sigma2, t, y)
                chats, covs = [], []
                for S in subsets:
                    _, c, C = _amplitude_posterior(J, h, tau2, sw2, S)
                    chats.append(c)
                    covs.append(C)
                F, qS, _ = _subset_posterior(J, h, tau2, sw2, rho, subsets)
                R = _residuals(Abar, y, subsets, qS, chats, covs, Kc, sub_idx)
                band = None if bands is None else bands[k]
                if FREQ_SIGMA0:
                    mu[k], sigma2[k] = _update_frequency_point(
                        mu[k], R[:, k], t, sw2, band)
                else:
                    mu[k], sigma2[k] = _update_frequency(mu[k], sigma2[k], R[:, k],
                                                         t, sw2, band)

        Abar, J, h = _expected_stats(mu, sigma2, t, y)
        chats, covs = [], []
        for S in subsets:
            _, c, C = _amplitude_posterior(J, h, tau2, sw2, S)
            chats.append(c)
            covs.append(C)
        F, qS, lse = _subset_posterior(J, h, tau2, sw2, rho, subsets)

        pi = np.array([sum(w for S, w in zip(subsets, qS) if k in S) for k in range(Kc)])
        for k in range(Kc):
            if pi[k] <= eps:
                continue
            _t2 = sum(
                w * (abs(c[S.index(k)]) ** 2 + np.real(C[S.index(k), S.index(k)]))
                for S, w, c, C in zip(subsets, qS, chats, covs) if k in S
            ) / pi[k]
            tau2[k] = _t2 if NOCLIP else max(_t2, eps)

        yn2 = float(np.real(y.conj() @ y))
        _sw2 = sum(
            w * (yn2
                 - 2.0 * np.real(c.conj() @ h[np.asarray(S)])
                 + np.real(np.trace((np.outer(c, c.conj()) + C)
                                    @ J[np.ix_(np.asarray(S), np.asarray(S))])))
            for S, w, c, C in zip(subsets, qS, chats, covs)
        ) / N
        sw2 = _sw2 if NOCLIP else max(_sw2, eps)

        _, J, h = _expected_stats(mu, sigma2, t, y)
        F, qS, lse = _subset_posterior(J, h, tau2, sw2, rho, subsets)
        # 点估计时略去高斯微分熵 0.5*log(sigma^2) (sigma^2=0 处发散, 且点估计
        # 的目标函数本就不含该项) —— 两种模式的 L **不可跨模式比较**。
        L = lse if FREQ_SIGMA0 else (lse + 0.5 * np.log(sigma2).sum())
        hist.append(float(L))
        if trace:
            # chats/covs/qS above are the ones this sweep's ELBO was scored with,
            # so the recorded amplitude is on exactly the final-output 口径
            # (AMP_MODEL_AVG=0 时同样退回 S*, 与末尾一致)。
            _js = int(np.argmax(F))
            _c, _ = _model_avg_amplitude(subsets, qS, chats, covs, Kc, eps,
                                         subsets[_js], chats[_js], covs[_js])
            mu_hist.append(mu.copy())
            amp_hist.append(np.abs(_c))
        if verbose:
            print(f"  sweep{sweep:>3d} elbo={hist[-1]: .6f} sw2={sw2:.3e}")
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * max(1.0, abs(hist[-1])):
            break

    Abar, J, h = _expected_stats(mu, sigma2, t, y)
    F, qS, _ = _subset_posterior(J, h, tau2, sw2, rho, subsets)
    pi = np.array([sum(w for S, w in zip(subsets, qS) if k in S) for k in range(Kc)])
    S_star = subsets[int(np.argmax(F))]
    _, c_star, C_star = _amplitude_posterior(J, h, tau2, sw2, S_star)
    c_map, v_map = _model_avg_amplitude(subsets, qS, chats, covs, Kc, eps,
                                        S_star, c_star, C_star)
    out_trace = {
        "mu": mu,
        "sigma": np.sqrt(sigma2),
        "tau2": tau2,
        "sigma_n2": sw2,
        "subsets": np.asarray(subsets, dtype=int),
        "qS": qS,
        "subset_scores": F,
        "pi": pi,
        "S_star": np.asarray(S_star, dtype=int),
        "amp_c": c_map,
        "amp": np.abs(c_map),
        "amp_var": v_map,
        "amp_sigma": np.sqrt(np.maximum(v_map, 0.0)),
        "score": pi * np.abs(c_map),
        "elbo": np.asarray(hist),
    }
    if trace:
        out_trace["mu_hist"] = np.asarray(mu_hist)
        out_trace["amp_hist"] = np.asarray(amp_hist)
    return out_trace


def _refine_one(args):
    """Top-level worker (must be importable for multiprocessing pickling)."""
    yb, tb, rb, mb, sb, K, n_sweeps, tol, b_bands, verbose, jacobi, trace = args
    return refine_stage2_tex_instance(yb, tb, rb, mb, sb, K, n_sweeps=n_sweeps,
                                      tol=tol, bands=b_bands, verbose=verbose,
                                      jacobi=jacobi, trace=trace)


# Box constraint on every mu_k, centred on its stage-1 candidate.  Without it the
# L-BFGS frequency step can walk a component that stage-1 had already localised
# out to a rotation sideband, with S_star unchanged -- a refinement failure, not a
# selection failure.
#
# The half-width is expressed as a MULTIPLE OF THE STAGE-1 BAND SPACING, not in
# absolute Hz: it has to admit the residual localisation error of the encoder
# (which scales with its grid step) while excluding the neighbouring sidebands.
# A value fixed in Hz does not transfer between encoders of different resolution
# -- measured at 20dB, a half-width of 1.0Hz gives 0.97x CRB on a 2Hz-grid
# checkpoint but 2.88x on a 1Hz-grid one, where 0.25Hz is needed instead.  Both
# points sit near band_spacing/2, which is the default below.
#
# BAND_FRAC=0 restores the unconstrained behaviour.  Callers pass the actual grid
# step via refine_stage2_tex_batch(grid_step_hz=...); band_hz= still overrides
# with an absolute value when a caller really wants one.
BAND_FRAC = float(os.environ.get("TEX_BAND_FRAC", 0.5))
# Fallback grid step used only when the caller supplies neither grid_step_hz nor
# band_hz.  Kept so old call sites keep working; prefer passing the real one.
BAND_GRID_HZ_DEFAULT = float(os.environ.get("TEX_BAND_GRID_HZ", 2.0))

# Frequency-step ordering: 1 = Jacobi (one refresh per sweep, as written in
# Algorithm 1 of stage2_derivation.tex), 0 = Gauss-Seidel (refresh after every
# coordinate, which is what this file did historically).
JACOBI = os.environ.get("TEX_JACOBI", "1").lower() in ("1", "true", "yes")

# TEX_NOCLIP=1 -> 去掉本文件里全部的范围限制:
#   * rho 的 clip(1e-6, 1-1e-6)  —— 它把编码器的硬否决 (logit=-inf) 降级成软惩罚
#   * sigma2 / tau2 / sigma_w^2 的初值下限 eps
#   * tau2 / sigma_w^2 每轮闭式更新的下限 eps  <- 最可疑: 会把参数塌陷压成病态数值
# 保留两处 (与 NOCLIP 无关, 恒定生效):
#   * amp_sigma 的 sqrt(max(v,0)): 负值纯粹来自浮点误差, 不是建模量
#   * _update_frequency 里 log sigma^2 的**两侧**边界: 见该函数 docstring ——
#     无下界 -> sigma^2 -> 0 -> log/除零 NaN; 无上界 -> 阻尼项下溢 -> mu 随机游走。
NOCLIP = os.environ.get("TEX_NOCLIP", "1").lower() in ("1", "true", "yes")

# ⭐ TEX_FREQ_SIGMA0=1: 把频率后验**退化成点估计** (sigma^2 恒 0), 只优化 mu。
#   ⚠ 不能简单地把 sigma2 设 0 再走原路径 —— ELBO 里的高斯微分熵 0.5*log(sigma^2)
#     在 sigma^2->0 时发散: _g_value 的 0.5*log(sigma2_k) -> -inf,
#     _g_grad 的 0.5/sigma2_k -> 除零, ELBO 的 0.5*np.log(sigma2).sum() -> -inf。
#     这不是数值实现问题, 是变分族本身: q(f)=N(mu,sigma^2) 退化成 delta 时
#     微分熵 -> -inf, sigma^2=0 是目标函数边界上的**奇点**。
#   ⭐ 正确做法 = 换目标函数: 点估计的目标里**本来就没有熵项**, 故这里
#     (a) 频率更新只对 mu 做一维优化, sigma2 恒 0;
#     (b) ELBO 汇总时略去 0.5*log(sigma2) 那一项 (见 refine 内的 L);
#     (c) 阻尼项 exp(-2*pi^2*sigma^2*t^2) 在 sigma^2=0 时干净地等于 1 (相干原子,
#         无退相干) —— 这一项**不需要**特殊处理, 原式直接成立。
#   ⚠ 输出的 "sigma" 字段将全为 0 ⇒ 任何除以 sigma 的下游 (校准/覆盖率) 都不适用。
FREQ_SIGMA0 = os.environ.get("TEX_FREQ_SIGMA0", "0").lower() in ("1", "true", "yes")

# ⭐ TEX_RHO_UNIFORM=1: 把候选先验 rho 全部置 0.5 (均匀先验), 方差照常优化。
#   rho 只经 _subset_score 的 log(rho/(1-rho)) 进入子集打分; rho=0.5 时该 logit
#   恒为 0, 且所有子集大小都是 K ⇒ logit.sum() 对每个子集都是同一个常数 0,
#   等价于**完全去掉编码器的候选先验**, 子集只由数据证据 (似然项) 排序。
#   ⚠ 与 FREQ_SIGMA0 正交: 这个开关不动 sigma, 用来单独隔离"先验的贡献"。
RHO_UNIFORM = os.environ.get("TEX_RHO_UNIFORM", "0").lower() in ("1", "true", "yes")

# 幅值输出口径, 见 refine_stage2_tex_instance 末尾:
#   1 (默认) = stage2_derivation.tex Outputs 写的 pi_k 加权模型平均
#   0        = 只取 MAP 子集 S* (本文件的历史行为)
# q(S) 退化成 one-hot 时两者恒等, 故差别只出现在低 SNR。
AMP_MODEL_AVG = os.environ.get("TEX_AMP_MODEL_AVG", "1").lower() in ("1", "true", "yes")


def refine_stage2_tex_batch(init_mu, init_logs, rho_prior, y_full, t_full, K,
                            n_sweeps=40, tol=1e-8, bands=None,
                            verbose=False, n_jobs=None, band_hz=None,
                            jacobi=None, grid_step_hz=None, trace=False):
    """Batch wrapper for refine_stage2_tex_instance.

    init_logs follows elbo_marginal.py convention: sigma = exp(init_logs).

    bands: explicit [B,Kc,2] (lo,hi) per candidate.  When left None, a symmetric
    box is built around init_mu with half-width BAND_FRAC * grid_step_hz (the
    stage-1 band spacing).  band_hz= overrides that with an absolute half-width in
    Hz; either set to 0 refines unconstrained.  See BAND_FRAC above for why the
    half-width is relative rather than a fixed number of Hz.

    trace=True adds per-instance "mu_hist"/"amp_hist" (per-sweep histories, ragged
    because of the tol break) next to "elbo"; see refine_stage2_tex_instance.

    Instances are independent, so they are dispatched to a process pool.
    n_jobs: None -> env TEX_JOBS, else min(cpu_count, B).  n_jobs=1 forces the
    serial path (identical results; the pool is only a scheduling change).
    """
    like = init_mu
    mu0 = _as_numpy(init_mu)
    sigma0 = np.exp(_as_numpy(init_logs))
    rho = _as_numpy(rho_prior)
    y = _as_numpy(y_full)
    t = _as_numpy(t_full)
    B = mu0.shape[0]
    if band_hz is not None:
        hw = float(band_hz)
    else:
        step = (BAND_GRID_HZ_DEFAULT if grid_step_hz is None else float(grid_step_hz))
        hw = BAND_FRAC * step
    if bands is not None:
        band_np = _as_numpy(bands)
    elif hw > 0:
        band_np = np.stack([mu0 - hw, mu0 + hw], axis=-1)
    else:
        band_np = None

    if n_jobs is None:
        n_jobs = int(os.environ.get("TEX_JOBS", 0)) or _DEFAULT_JOBS
    n_jobs = max(1, min(int(n_jobs), B))

    jac = JACOBI if jacobi is None else bool(jacobi)
    tasks = [(y[b], t[b], rho[b], mu0[b], sigma0[b], K, n_sweeps, tol,
              None if band_np is None else band_np[b], verbose, jac, trace)
             for b in range(B)]

    if n_jobs == 1:
        outs = [_refine_one(a) for a in tasks]
    else:
        # BLAS threads must be pinned to 1 inside workers: each instance is a
        # small dense solve, so nested threading oversubscribes the box and is
        # slower than the serial path.
        with ProcessPoolExecutor(max_workers=n_jobs,
                                 initializer=_init_worker) as ex:
            outs = list(ex.map(_refine_one, tasks, chunksize=1))

    keys = ["mu", "sigma", "tau2", "pi", "amp", "amp_c", "amp_var",
            "amp_sigma", "score", "S_star"]
    out = {k: _maybe_tensor(np.stack([o[k] for o in outs], axis=0), like) for k in keys}
    out["sigma_n2"] = _maybe_tensor(np.asarray([o["sigma_n2"] for o in outs]), like)
    out["qS"] = _maybe_tensor(np.stack([o["qS"] for o in outs], axis=0), like)
    out["subset_scores"] = _maybe_tensor(np.stack([o["subset_scores"] for o in outs], axis=0), like)
    out["subsets"] = _maybe_tensor(outs[0]["subsets"], like)
    out["elbo"] = [o["elbo"] for o in outs]
    if trace:
        # ragged across instances (tol break), so kept as per-instance lists
        out["mu_hist"] = [o["mu_hist"] for o in outs]
        out["amp_hist"] = [o["amp_hist"] for o in outs]
    return out
