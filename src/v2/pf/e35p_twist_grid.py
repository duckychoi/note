"""Equal-budget comparison where every method picks its best depth/width split (n steps x K particles) - #35 policy A."""
import os, json, numpy as np
import e35p_twist as E

def main():
    out = {}
    for d in [2, 8, 32]:
        P = E.setup(d, lam=0.15, beta=1.0)
        print(f"\nd={d}  (mode-mass L1 to exact tilted target; best split shown as n x K)")
        for B in [32, 128, 512]:
            line = f" B={B:4d}"
            for name in ["bon", "guid_heur", "fk_heur", "tds_heur", "guid_exact", "tds_exact"]:
                per = 2 if name.startswith(("guid", "tds")) else 1
                ns = [B // 2] if name.startswith("guid") else [n for n in [4, 8, 16, 32, 64] if B // (n * per) >= (1 if name == "bon" else 2)]
                best = None
                for n in ns:
                    K = B // (n * per)
                    if name == "bon": fn = lambda r: E.bon(P, B, n, r)
                    elif name == "guid_heur": fn = lambda r: E.smc(P, B, n, r, "heur", guide=True, weights=False)
                    elif name == "guid_exact": fn = lambda r: E.smc(P, B, n, r, "exact", guide=True, weights=False)
                    elif name == "fk_heur": fn = lambda r: E.smc(P, B, n, r, "heur")
                    elif name == "tds_heur": fn = lambda r: E.smc(P, B, n, r, "heur", guide=True)
                    else: fn = lambda r: E.smc(P, B, n, r, "exact", guide=True)
                    ml = np.mean([E.evaluate(P, fn(np.random.default_rng(10 * d + B + s)))[0] for s in range(3)])
                    out[f"d{d}/B{B}/{name}/n{n}"] = float(ml)
                    if best is None or ml < best[0]: best = (ml, n, K)
                line += f" | {name} {best[0]:.3f} ({best[1]}x{best[2]})"
            print(line, flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
    json.dump(out, open(os.path.join(od, "e35p_twist_grid.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
