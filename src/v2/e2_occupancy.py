"""
E2 - does "max-ent RL + exact log-likelihood reward" match the expert distribution in an MDP?

Ergodic deterministic MDP (ring, N states, actions {-1,0,+1}); a 'motion window' = transition (s,s').
Expert p(s,s') = stationary transition distribution of a random expert policy (realizable target).
Exact stationary distribution d_pi(s) (differentiable linear solve), exact gradients (torch, float64).

  (A) max-ent RL, reward r = log p(s,s'), alpha=1        J = E_d log p + E_d H(pi(.|s))
      Claim:  J = -KL(d||p) - H(d_S) + H(A|S,S')  (last term 0 if injective; -H(d_S) still biases A — only the corrected C is unbiased when that term is constant) -> biased toward LOW state entropy
  (B) learner-density-corrected reward r = log p(s,s') - log d_pi(s,s')   J = -KL(d||p)
      (this is what a second, learner-fitted flow supplies; GAIL/SMILING-style)
  (C) (A) + state-marginal correction r = log p - log d_pi(s)   == (B) only if action->next-state is injective per state
      (general: J_C = -KL - 0 + H(A|S,S');  --noninj adds a duplicate 'stay' action to break injectivity)
  (D) next-state entropy H(S'|S) (actions with the same destination summed) + state correction == -KL in any
      deterministic finite MDP
Outputs: results/v2/e2_occupancy.json
"""
import json, os
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "v2")

import sys
N = 12
# injective: actions {-1,0,+1};  non-injective (codex C6 counter-case): {-1,0,0,+1} (two actions -> same s')
MOVES = torch.tensor([-1, 0, 0, 1]) if "--noninj" in sys.argv else torch.tensor([-1, 0, 1])
A = len(MOVES)
NXT = (torch.arange(N)[:, None] + MOVES[None, :]) % N        # (N,A)
ONEHOT = torch.nn.functional.one_hot(NXT, N).double()        # (N,A,N): action -> next state


def stationary(logits):
    pi = torch.softmax(logits, -1)                             # (N,A)
    P = torch.zeros(N, N)
    P = P.index_put((torch.arange(N)[:, None].expand(N, A), NXT), pi, accumulate=True)
    M = P.T - torch.eye(N)
    M = torch.cat([M[:-1], torch.ones(1, N)], 0)
    b = torch.zeros(N); b[-1] = 1
    ds = torch.linalg.solve(M, b)
    return ds, pi


def trans_dist(logits):
    """returns d(s,s') (N,N), d(s), pi.  d(s,s') sums actions with the same destination."""
    ds, pi = stationary(logits)
    Pss = torch.einsum("sa,san->sn", pi, ONEHOT)
    return ds[:, None] * Pss, ds, pi


def exact_A(p):
    """Exact optimum of (A): average-reward soft RL with r(s,a)=log p(s,f(s,a)), alpha=1.
    M_ss' = m_ss' p(s,s') (m = #actions reaching s'),  J* = log rho(M),
    pi*(a|s) ∝ p(s,f(s,a)) v(f(s,a))  (v = Perron right eigenvector)."""
    m = ONEHOT.sum(1)                                           # (N,N) action multiplicity
    Mx = (m * p).numpy()
    w, V = np.linalg.eig(Mx); i = np.argmax(w.real); rho = w[i].real; v = np.abs(V[:, i].real)
    Pss = Mx * v[None, :] / (rho * v[:, None])                  # transition under pi*
    w2, U = np.linalg.eig(Pss.T); j = np.argmin(np.abs(w2 - 1)); ds = np.abs(U[:, j].real); ds /= ds.sum()
    d = ds[:, None] * Pss
    msk = d > 0
    kl = float((d[msk] * (np.log(d[msk]) - np.log(p.numpy()[msk]))).sum())
    return float(np.log(rho)), kl, float(-(ds * np.log(ds)).sum())


def run(seed, steps=4000):
    g = torch.Generator().manual_seed(seed)
    exp_logits = 2.0 * torch.randn(N, A, generator=g)
    p, ps, _ = trans_dist(exp_logits)
    p = p.detach(); logp = torch.log(p.clamp_min(1e-300))
    Hps = float(-(ps * ps.log()).sum())
    res = {}
    for mode in MODES:
        th = torch.zeros(N, A, requires_grad=True)
        opt = torch.optim.Adam([th], lr=0.05)
        for _ in range(steps):
            d, ds, pi = trans_dist(th)
            Hpi = -(pi * torch.log(pi + 1e-300)).sum(-1)
            if mode == "A_maxent_logp":
                J = (d * logp).sum() + (ds * Hpi).sum()
            elif mode == "B_logp_minus_logd":
                J = (d * (logp - torch.log(d + 1e-300))).sum()
            elif mode == "C_maxent_plus_state_corr":
                J = (d * logp).sum() + (ds * Hpi).sum() - (ds * torch.log(ds + 1e-300)).sum()
            else:   # D: next-state entropy H(S'|S) instead of action entropy (general deterministic MDP)
                Pss = d / ds[:, None]
                Hn = -(Pss * torch.log(Pss + 1e-300)).sum(-1)
                J = (d * logp).sum() + (ds * Hn).sum() - (ds * torch.log(ds + 1e-300)).sum()
            opt.zero_grad(); (-J).backward(); opt.step()
        d, ds, pi = trans_dist(th.detach())
        m = d > 0
        kl = float((d[m] * (torch.log(d[m]) - logp[m])).sum())
        Hds = float(-(ds * ds.log()).sum())
        res[mode] = dict(KL=kl, H_state=Hds)
        if mode == "A_maxent_logp":
            with torch.no_grad():
                dA, dsA, piA = trans_dist(th.detach())
                HpiA = -(piA * torch.log(piA + 1e-300)).sum(-1)
                J_adam = float((dA * logp).sum() + (dsA * HpiA).sum())
            J_star, kl_star, H_star = exact_A(p)
            res[mode].update(J_adam=J_adam, J_star=J_star, J_gap=J_star - J_adam, KL_exact=kl_star, H_state_exact=H_star)
    res["H_state_expert"] = Hps
    return res


MODES = ["A_maxent_logp", "B_logp_minus_logd", "C_maxent_plus_state_corr", "D_nextstate_ent_plus_state_corr"]
if __name__ == "__main__":
    allr = [run(s) for s in range(10)]
    print("non-injective" if "--noninj" in sys.argv else "injective", "MDP; success tolerance KL < 1e-4")
    print(f"{'mode':28s} {'KL(d||p) mean±sd':>20s} {'H(d_S)-H(p_S) mean':>20s}")
    summ = {}
    for m in MODES:
        kl = np.array([r[m]["KL"] for r in allr])
        dh = np.array([r[m]["H_state"] - r["H_state_expert"] for r in allr])
        if m == "A_maxent_logp":
            ke = np.array([r[m]["KL_exact"] for r in allr]); gp = np.array([r[m]["J_gap"] for r in allr])
            he = np.array([r[m]["H_state_exact"] - r["H_state_expert"] for r in allr])
            print(f"{'  A exact optimum (eigvec)':32s} KL {ke.mean():.3e} ± {ke.std():.2e} (min {ke.min():.3e})  dH {he.mean():+.4f}"
                  f"  | Adam J-gap mean {gp.mean():.2e} max {gp.max():.2e}")
        summ[m] = dict(KL_mean=kl.mean(), KL_sd=kl.std(), KL_max=kl.max(), tol=1e-4, n_ok=int((kl < 1e-4).sum()),
                       n_seeds=len(kl), dH_mean=dh.mean(), dH_sd=dh.std())
        if m == "A_maxent_logp":
            summ[m].update(n_ok_exact=int((ke < 1e-4).sum()), KL_exact_mean=ke.mean(), KL_exact_sd=ke.std(), KL_exact_min=ke.min(), dH_exact_mean=he.mean(),
                           J_gap_mean=gp.mean(), J_gap_max=gp.max())
        print(f"{m:32s} KL {kl.mean():.3e} ± {kl.std():.2e} (max {kl.max():.2e}, ok {int((kl<1e-4).sum())}/10)"
              f"  dH {dh.mean():+.4f} ± {dh.std():.4f}")
    json.dump(dict(per_seed=allr, summary=summ), open(os.path.join(OUT, "e2_occupancy" + ("_noninj" if "--noninj" in sys.argv else "") + ".json"), "w"), indent=1)
