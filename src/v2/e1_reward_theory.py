"""
E1 - exact check of the FM-residual reward theory (1-D, quadrature, no estimators).

Claims tested
  T1  KL(q||p) == B_w(q) := int_0^1 t/(1-t) E_{x~q_t} |u_q - u_p|^2 dt        (identity, not a bound)
      vs the uniform-weight bias B_u(q) used in docs/05 (w=1).
  T2  per-sample likelihood-weighted residual  l_w(x) = int_0^{1-delta} t/(1-t) E_x0 |u_p(x_t)-(x-x0)|^2 dt
      satisfies  l_w(x) = KL(N(x,tau)||p_tau) = -log p(x) + K_delta + o(1);  g = exp(-(l_w-K_delta)) has int g <= 1,
      so the max-ent (alpha=1, entropy of the SAMPLE distribution) Gibbs density is q* = g / int g -> p
      while the uniform residual l_u(x) (= the boxed reward of the concept note) does NOT:
      q*_u ∝ exp(-l_u/alpha) is biased for EVERY alpha.
Outputs: results/v2/e1_reward_theory.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from gmm_flow import GMM, gl, gh

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "v2")

XG = np.linspace(-25, 25, 10001)
DX = 50.0 / 10000


def kl_grid(lq, lp):
    q = np.exp(lq); q /= q.sum() * DX
    lq = np.log(q + 1e-300)
    return float(((lq - lp) * q).sum() * DX)


def bias(q, p, weight, nt=400, smax=12.0, xg=None):
    """int_0^1 weight(t) E_{q_t}|u_q-u_p|^2 dt, with t = 1-exp(-s) to resolve t->1."""
    s, ws = gl(0.0, smax, nt)                 # t up to 1-exp(-smax)
    t = 1 - np.exp(-s); dt = np.exp(-s) * ws  # dt = (1-t) ds
    tot = 0.0
    xg = XG if xg is None else xg
    X = xg[:, None]; dx = (xg[-1] - xg[0]) / (len(xg) - 1)
    for ti, dti in zip(t, dt):
        if ti < 1e-9:
            continue
        qt = np.exp(q.logpdf_t(X, ti))
        du = (q.velocity(X, ti) - p.velocity(X, ti))[:, 0]
        tot += weight(ti) * dti * (qt * du**2).sum() * dx
    return float(tot)


w_like = lambda t: t / (1 - t)
w_unif = lambda t: 1.0

# ----------------------------------------------------------------- T1
P = GMM([0.6, 0.3, 0.1], [[-2.5], [2.5], [0.0]], [0.55, 0.55, 0.4])
QS = {
    "shift+0.5":      GMM([0.6, 0.3, 0.1], [[-2.0], [3.0], [0.5]], [0.55, 0.55, 0.4]),
    "reweight":       GMM([0.2, 0.3, 0.5], [[-2.5], [2.5], [0.0]], [0.55, 0.55, 0.4]),
    "single-mode":    GMM([1.0], [[-2.5]], [0.55]),
    "too-wide(x2)":   GMM([0.6, 0.3, 0.1], [[-2.5], [2.5], [0.0]], [1.1, 1.1, 0.8]),
    "too-narrow(/3)": GMM([0.6, 0.3, 0.1], [[-2.5], [2.5], [0.0]], [0.18, 0.18, 0.13]),
    "gaussian-fit":   GMM([1.0], [[-0.75]], [2.3]),
    "point-like":     GMM([1.0], [[-2.5]], [0.05]),
}
lp = P.logpdf(XG[:, None])
# ----------------------------------------------------------------- T2 helpers
Z0, W0 = gh(120)


def per_sample(p, x, weight, delta, ns=240):
    """l(x) = int_0^{1-delta} weight(t) E_x0 |u_p(x_t,t)-(x-x0)|^2 dt  (1-D)."""
    s, ws = gl(0.0, -np.log(delta), ns)
    t = 1 - np.exp(-s); dt = np.exp(-s) * ws
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for ti, dti in zip(t, dt):
        Xt = (1 - ti) * Z0[None, :] + ti * x[:, None]            # (n,m)
        u = p.velocity(Xt.reshape(-1, 1), ti).reshape(Xt.shape)    # stable component-wise velocity
        res2 = (u - (x[:, None] - Z0[None, :])) ** 2
        out += weight(ti) * dti * (res2 * W0[None, :]).sum(1)
    return out


xs = np.linspace(-6, 6, 1201); dxs = 12.0 / 1200
lp_s = P.logpdf(xs[:, None]); p_s = np.exp(lp_s); p_s /= p_s.sum() * dxs


def kl_s(lq):
    """KL(q||p) = int p [r log r - (r-1)], r = q/p; with x = log r the integrand is x e^x - expm1(x) (series if small)."""
    q = np.exp(lq - lq.max()); q /= q.sum() * dxs
    x = np.log(q + 1e-300) - np.log(p_s + 1e-300)
    f = np.where(np.abs(x) < 1e-3, x**2 / 2 + x**3 / 3 + x**4 / 8, x * np.exp(x) - np.expm1(x))
    return float((p_s * f).sum() * dxs)


def wstd(a):  # p-weighted std of a (how far from "constant + log p")
    mu = (a * p_s).sum() * dxs
    return float(np.sqrt((((a - mu) ** 2) * p_s).sum() * dxs))




def main():
    global Z0, W0
    from scipy.optimize import minimize_scalar
    os.makedirs(OUT, exist_ok=True)
    t1 = []
    print("T1  KL(q||p) vs likelihood-weighted bias B_w and uniform bias B_u (docs/05)")
    print(f"{'q':16s} {'KL':>9s} {'B_w':>9s} {'B_w/KL':>7s} {'B_u':>9s} {'B_u/KL':>7s}")
    for name, Q in QS.items():
        kl = kl_grid(Q.logpdf(XG[:, None]), lp)
        bw = bias(Q, P, w_like)
        bu = bias(Q, P, w_unif)
        t1.append(dict(q=name, KL=kl, B_w=bw, B_u=bu))
        print(f"{name:16s} {kl:9.4f} {bw:9.4f} {bw/kl:7.3f} {bu:9.4f} {bu/kl:7.3f}")

    # convergence of T1, one knob at a time (codex r3 C7)
    base = dict(nt=800, smax=16.0)
    ref = [bias(Q, P, w_like, **base) for Q in QS.values()]
    def dmax(vals):
        return max(abs(a - b) for a, b in zip(vals, ref))
    XG_range = np.linspace(-30, 30, 12001)     # same spacing 0.005, wider range
    XG_fine = np.linspace(-25, 25, 20001)      # same range, half spacing
    T1_conv = dict(reference="nt=800, s_max=16, x in [-25,25] 10001 pts")
    T1_conv["nodes_400_vs_800_at_s16"] = dmax([bias(Q, P, w_like, nt=400, smax=16.0) for Q in QS.values()])
    T1_conv["smax_12_vs_16_at_800"] = dmax([bias(Q, P, w_like, nt=800, smax=12.0) for Q in QS.values()])
    T1_conv["smax_20_vs_16_at_800"] = dmax([bias(Q, P, w_like, nt=800, smax=20.0) for Q in QS.values()])
    T1_conv["space_range_30_vs_25_same_dx"] = dmax([bias(Q, P, w_like, xg=XG_range, **base) for Q in QS.values()])
    T1_conv["space_dx_0.0025_vs_0.005_same_range"] = dmax([bias(Q, P, w_like, xg=XG_fine, **base) for Q in QS.values()])
    T1_conv["max_gap_KL_vs_Bw_reference"] = max(abs(r["KL"] - b) for r, b in zip(t1, ref))
    for kk, vv in T1_conv.items():
        print(f"  T1 convergence {kk}: {vv}")
    t2 = {}
    print("\nT2  per-sample reward vs log p  (fixed expert flow, no learner model)")
    for delta in [1e-2, 1e-3, 1e-4]:
        lw = per_sample(P, xs, w_like, delta)
        gap = lw + lp_s                                   # should be ~constant
        klw = kl_s(-lw)
        # exact per-sample identity (codex C4): l_w(x) = KL(N(x,tau)||p_tau), tau=(delta/(1-delta))^2
        tau = (delta / (1 - delta)) ** 2
        Pt = GMM(P.w, P.mu, np.sqrt(P.s**2 + tau))
        zz, ww = gh(80)
        exact = np.array([-0.5*np.log(2*np.pi*np.e*tau) - (ww*Pt.logpdf((x+np.sqrt(tau)*zz)[:, None])).sum() for x in xs])
        err = float(np.abs(exact - lw).max())
        t2[f"like_delta={delta}"] = dict(std_gap_under_p=wstd(gap), KL_fixpoint=klw, max_abs_err_vs_exact_formula=err)
        print(f"  likelihood-weight  delta={delta:.0e}: std_p[l_w+log p]={wstd(gap):.2e}  KL(q*_w||p)={klw:.2e} (q*_w = exp(-l_w)/Z)"
              f"  |l_w - KL(N(x,tau)||p_tau)|max={err:.2e}")
    lu = per_sample(P, xs, w_unif, 1e-6)
    from scipy.optimize import minimize_scalar
    os.makedirs(OUT, exist_ok=True)
    la_grid = np.linspace(np.log(0.05), np.log(20), 241)
    kl_a = np.array([kl_s(-lu / np.exp(la)) for la in la_grid])
    i0 = int(np.argmin(kl_a)); lo, hi = la_grid[max(i0 - 1, 0)], la_grid[min(i0 + 1, len(la_grid) - 1)]
    opt = minimize_scalar(lambda la: kl_s(-lu / np.exp(la)), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-8))
    if not opt.success:
        raise RuntimeError(f"alpha search failed: {opt.message}")
    best = (float(np.exp(opt.x)), float(opt.fun))
    alpha_search = dict(method="241-pt grid on log alpha in [log .05, log 20], then bounded refine in the bracketing cell",
                        success=bool(opt.success), message=str(opt.message), grid_best_alpha=float(np.exp(la_grid[i0])),
                        n_local_minima_on_grid=int(((kl_a[1:-1] < kl_a[:-2]) & (kl_a[1:-1] < kl_a[2:])).sum()))
    # alpha required to reproduce the density ratio between two points: alpha = (l_u(b)-l_u(a)) / (log p(a)-log p(b))
    pts = np.array([-2.5, 2.5, 0.0]); lu_p = per_sample(P, pts, w_unif, 1e-6); lp_p = P.logpdf(pts[:, None])
    req = {f"{pts[a]}vs{pts[b]}": float((lu_p[b]-lu_p[a])/(lp_p[a]-lp_p[b])) for a, b in [(0, 1), (0, 2), (1, 2)]}
    t2["uniform_required_alpha_pairs"] = req
    # Prop 3 sign: vary truncation, time nodes, Gauss-Hermite order (numerical observation, not an error bound)
    rob = {}
    for tag, dl, ns, ng in [("base", 1e-6, 240, 120), ("delta1e-8", 1e-8, 240, 120), ("ns480", 1e-6, 480, 120), ("gh200", 1e-6, 240, 200)]:
        Z0, W0 = gh(ng)
        v = per_sample(P, pts, w_unif, dl, ns=ns)
        rob[tag] = dict(dlu_2p5_minus_0=float(v[1] - v[2]), required_alpha=float((v[1] - v[2]) / (lp_p[2] - lp_p[1])))
    Z0, W0 = gh(120)
    t2["prop3_sign_robustness"] = rob
    print("  Prop3 l_u(2.5)-l_u(0) under quadrature changes:", {kk: round(vv["dlu_2p5_minus_0"], 9) for kk, vv in rob.items()})
    print(f"  uniform: alpha needed to match density ratios of pairs (one alpha cannot fit all): {req}")
    # best affine fit: is -l_u an affine function of log p at all?
    A = np.vstack([lp_s, np.ones_like(lp_s)]).T
    coef, *_ = np.linalg.lstsq(A * np.sqrt(p_s)[:, None], -lu * np.sqrt(p_s), rcond=None)
    resid = -lu - A @ coef
    t2["grid_T2"] = "x in [-6,6], 1201 pts; densities renormalised on this interval"
    t2["uniform"] = dict(KL_fixpoint_alpha1=kl_s(-lu), best_alpha=best[0], KL_fixpoint_best_alpha=best[1],
                         alpha_search=alpha_search,
                         affine_fit_slope=float(coef[0]), std_resid_under_p=wstd(resid))
    print(f"  uniform (concept-note reward): KL(q*_u||p)={kl_s(-lu):.4f} at alpha=1 (q*_u = exp(-l_u/alpha)/Z);"
          f" best alpha within searched range [0.05,20]={best[0]:.4f} -> KL={best[1]:.4f}")
    print(f"    -l_u ~ {coef[0]:.3f}*log p + c  with p-weighted resid std {wstd(resid):.4f}")

    # mode masses of fixed points (what the notes call 'mode coverage')
    def masses(lq):
        """window mass with explicit end points (trapezoid on grid + interpolated ends)."""
        q = np.exp(lq - lq.max()); q /= np.trapezoid(q, xs)
        out = []
        for c in (-2.5, 2.5, 0.0):
            a, b = c - 1.2, c + 1.2
            m = (xs > a) & (xs < b)
            xx = np.concatenate([[a], xs[m], [b]])
            qq = np.concatenate([[np.interp(a, xs, q)], q[m], [np.interp(b, xs, q)]])
            out.append(float(np.trapezoid(qq, xx)))
        return out
    from scipy.stats import norm
    def masses_target_cdf():
        return [float(sum(w * (norm.cdf(c + 1.2, m, sd) - norm.cdf(c - 1.2, m, sd)) for w, m, sd in zip(P.w, P.mu[:, 0], P.s)))
                for c in (-2.5, 2.5, 0.0)]
    lw3 = per_sample(P, xs, w_like, 1e-4)
    t2["mode_mass"] = dict(target_same_window=masses(lp_s), target_exact_cdf=masses_target_cdf(), like=masses(-lw3), unif_alpha1=masses(-lu),
                           unif_best=masses(-lu / best[0]))
    print(f"  mode mass (+-1.2 windows) target {np.round(masses(lp_s),3)} | like {np.round(masses(-lw3),3)} | "
          f"unif a=1 {np.round(masses(-lu),3)} | unif best-a {np.round(masses(-lu/best[0]),3)}")

    json.dump(dict(T1=t1, T1_convergence=T1_conv, T2=t2), open(os.path.join(OUT, "e1_reward_theory.json"), "w"), indent=1)
    print("saved", os.path.abspath(os.path.join(OUT, "e1_reward_theory.json")))


if __name__ == "__main__":
    main()
