"""#35 policy candidate B: does a more accurate tilted sampler (SMC) exploit a value OVER-estimate more than best-of-N?
Q_hat = Q + H * bump at an out-of-distribution action (behaviour density ~1e-9 there). True metrics use the true Q.
LCB fix: ensemble of 5 Q_hat with bump heights H_i ~ U(0, 2H) at the same OOD spot -> mean - 1*std.
Reports: OOD mass (fraction within 1.0 of the bump) and true E[Q] (higher better)."""
import numpy as np, e35p_twist as E

def run():
    d = 8
    P = E.setup(d, lam=0.15, beta=1.0)
    loc = np.zeros(d)                      # gap between behaviour modes (low but nonzero density)
    trueP = dict(P)
    rng = np.random.default_rng(0); Hs = rng.uniform(0, 2 * 10.0, 5)
    print("bump heights", Hs.round(2), "mean", Hs.mean().round(2), "LCB(mean-std)", (Hs.mean() - Hs.std()).round(2))
    for label, H in [("true Q", 0.0), ("Q_hat (H=10)", 10.0), ("LCB", Hs.mean() - Hs.std())]:
        P2 = dict(P); P2["bumps"] = [(loc, H, 0.6)] if H > 0 else []
        line = f"{label:12s}"
        for B in [32, 128, 512]:
            for name, fn in [("bon", lambda r: E.bon(P2, B, 8, r)), ("fk", lambda r: E.smc(P2, B, 16, r, "heur")),
                             ("guid", lambda r: E.smc(P2, B, B // 2, r, "heur", guide=True, weights=False))]:
                X = np.concatenate([fn(np.random.default_rng(B + s)) for s in range(2)])
                ood = (((X - loc) ** 2).sum(-1) < 1.0).mean(); tq = E.Q(trueP, X).mean()
                line += f" | B{B} {name} ood {ood:.3f} Q {tq:.2f}"
        print(line, flush=True)

if __name__ == "__main__":
    run()
