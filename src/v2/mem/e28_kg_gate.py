"""
#28 gate v2, stage 2 (pre-registered in the research note §3): can a SUPERPOSED holographic memory, queried by chaining
relations, answer multi-hop questions whose answer was never stored, from OLD facts, better than explicit fact buffers
of the same byte budget?

Graph: N=400 entities in 40 clusters of 10; R=3 base relations; every entity has k=3 r-successors drawn inside the
cluster shifted by an r-specific offset (so 2-hop paths r1->r2 from h converge on a few entities with different path
counts).  Derived query (h, r1, r2): answer = entity with the most r1->r2 paths (ties -> any of the tied set counts);
derived facts are never stored.  Ordered pairs r1 != r2 (order matters: r1->r2 and r2->r1 have different answers).
Stream (v2 after smoke test): the 3600 query-graph facts in random order, THEN 3600 distractor facts of a second graph
on 400 other entities (same structure).  Every query is therefore "old": all its supporting facts are >= 3600 steps old.
holo_raw  : chaining WITHOUT clean-up (crosstalk multiplies across hops; smoke test: 0.0 at D=2048)
holo_clean: after hop 1, project onto the entity codebook, keep entities with score > tau (soft weights), rebuild the
            superposition, then hop 2  -> still sums all paths in one vector, no path enumeration.
            The entity codebook is regenerated from a seed (not counted in the budget; buffers likewise store ids only).

Methods (budget in bytes):
  holo  : per relation M_r = sum_facts lam^age * conj(h) * Pi(t)  (D complex, Pi = cyclic shift keeps hop order),
          query v1 = Pi^-1(q * M_r1), v2 = Pi^-1(v1 * M_r2), answer = argmax_e Re<v2, e>.  No path enumeration, no clean-up.
          bytes = R*D*b/8 with b bits per complex component: b=32 (fp16 re/im), b=8 (4-bit magnitude + 4-bit phase),
          b=2 (2-bit phase only).  Quantisation applied to the final state (streaming re-quantisation NOT tested).
  fifo  : keep the most recent C facts, exact path counting on the kept subgraph.     C = bytes / 6 (3 x int16 per fact)
  resv  : reservoir sample of C facts (uniform over the whole stream), exact path counting.
  oracle: full graph (upper bound, = 1.0 by construction).
Decay lam = exp(-alpha) picked on validation streams; report evaluation streams (5 seeds, mean +- 95% t-interval).
Pass (pre-registered): holo beats BOTH fifo and resv on OLD queries at equal bytes, CI-separated, at some budget.
Results -> results/v2/e28_kg_gate.json
"""
import os, json, itertools
import numpy as np
from scipy import stats

N, CL, R, K = 400, 40, 3, 3
OFF = [3, 11, 24]


def make_graph(rng, base=0):
    cl = np.arange(N) // (N // CL)
    succ = np.zeros((R, N, K), int)
    for r in range(R):
        for h in range(N):
            c2 = (cl[h] + OFF[r]) % CL
            succ[r, h] = rng.choice(np.arange(c2 * 10, c2 * 10 + 10), K, replace=False)
    facts = [(h + base, r, int(t) + base) for r in range(R) for h in range(N) for t in succ[r, h]]
    order = rng.permutation(len(facts)); facts = [facts[i] for i in order]
    return facts


def path_counts(facts_subset, h, r1, r2):
    nb = {}
    for (a, r, b) in facts_subset:
        nb.setdefault((a, r), []).append(b)
    cnt = np.zeros(N)
    for x in nb.get((h, r1), []):
        for t in nb.get((x, r2), []):
            cnt[t] += 1
    return cnt


def queries(facts, rng, n=300):
    T = len(facts); arr = {f: i for i, f in enumerate(facts)}
    full = facts; out = []
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        h = int(rng.integers(N)); r1, r2 = rng.choice(R, 2, replace=False)
        cnt = path_counts_cache(full, h, r1, r2)
        if cnt.max() < 2:
            continue
        best = set(np.where(cnt == cnt.max())[0].tolist())
        # supporting facts of the winning paths
        sup = [arr[(h, int(r1), x)] for x in NB[(h, int(r1))] for t in NB.get((x, int(r2)), []) if t in best] + \
              [arr[(x, int(r2), t)] for x in NB[(h, int(r1))] for t in NB.get((x, int(r2)), []) if t in best]
        out.append(dict(h=h, r1=int(r1), r2=int(r2), best=best, old=True))
    return out


NB = {}


def path_counts_cache(facts, h, r1, r2):
    global NB
    if not NB:
        for (a, r, b) in facts:
            NB.setdefault((a, r), []).append(b)
    cnt = np.zeros(N)
    for x in NB.get((h, r1), []):
        for t in NB.get((x, r2), []):
            cnt[t] += 1
    return cnt


def quantize(M, bits):
    if bits == 32:
        return (M.real.astype(np.float16) + 1j * M.imag.astype(np.float16)).astype(np.complex64)
    ph = np.angle(M)
    if bits == 2:
        return np.exp(1j * (np.round(ph / (np.pi / 2)) * (np.pi / 2)))
    mag = np.abs(M); lm = np.log(mag + 1e-9); lo, hi = np.percentile(lm, 1), np.percentile(lm, 99)
    qm = np.exp(lo + np.round(np.clip((lm - lo) / (hi - lo + 1e-9), 0, 1) * 15) / 15 * (hi - lo))
    return qm * np.exp(1j * np.round(ph / (2 * np.pi / 16)) * (2 * np.pi / 16))


def holo_answer(facts, E, D, alpha, bits, Q, clean=False, tau=0.3):
    T = len(facts); M = np.zeros((R, D), complex); lam = np.exp(-alpha)
    for i, (h, r, t) in enumerate(facts):
        M[r] += lam ** (T - 1 - i) * np.conj(E[h]) * np.roll(E[t], 1)
    M = np.stack([quantize(M[r], bits) for r in range(R)])
    hits = []
    for q in Q:
        v1 = np.roll(E[q["h"]] * M[q["r1"]], -1)
        if clean:
            w = np.real(np.conj(E) @ v1) / D; w = np.where(w > tau * w.max(), w, 0.0); v1 = w @ E
        v2 = np.roll(v1 * M[q["r2"]], -1)
        hits.append(int(np.argmax(np.real(np.conj(E[:N]) @ v2))) in q["best"])
    return np.array(hits)


def buffer_answer(facts, C, kind, rng, Q):
    if kind == "fifo":
        keep = facts[-C:]
    else:
        idx = rng.choice(len(facts), min(C, len(facts)), replace=False); keep = [facts[i] for i in idx]
    nb = {}
    for (a, r, b) in keep:
        nb.setdefault((a, r), []).append(b)
    hits = []
    for q in Q:
        cnt = np.zeros(N)
        for x in nb.get((q["h"], q["r1"]), []):
            for t in nb.get((x, q["r2"]), []):
                cnt[t] += 1
        pred = int(np.argmax(cnt + 1e-6 * rng.random(N))) if cnt.max() > 0 else int(rng.integers(N))
        hits.append(pred in q["best"])
    return np.array(hits)


def run_seed(seed, D, bits, alpha):
    global NB
    rng = np.random.default_rng(seed); NB = {}
    facts = make_graph(rng); Q = queries(facts, rng); facts = facts + make_graph(rng, base=N)
    E = np.exp(1j * rng.uniform(-np.pi, np.pi, (2 * N, D)))
    old = np.array([q["old"] for q in Q])
    C = int(R * D * bits / 8 / 6)
    res = {"holo_raw": holo_answer(facts, E, D, alpha, bits, Q), "holo": holo_answer(facts, E, D, alpha, bits, Q, clean=True), "fifo": buffer_answer(facts, C, "fifo", rng, Q),
           "resv": buffer_answer(facts, C, "resv", rng, Q)}
    return {k: (float(v.mean()), 0.0) for k, v in res.items()}, C, len(Q)


def main():
    out = {}
    print(f"query facts {N*R*K} then {N*R*K} distractors; budget bytes = R*D*bits/8; buffer capacity C = bytes/6 facts")
    for bits in [32, 8, 2]:
        for D in [512, 2048, 8192]:
            # pick alpha on validation seeds 100,101
            best = max([0.0, 1e-4, 5e-4, 2e-3], key=lambda a: np.mean([run_seed(100 + s, D, bits, a)[0]["holo"][0] for s in range(2)]))
            rs = [run_seed(s, D, bits, best) for s in range(5)]
            C = rs[0][1]; line = f"bits {bits:2d} D {D:5d} (bytes {R*D*bits//8:6d}, C={C:5d} facts, alpha*={best}) | OLD queries:"
            rec = {}
            for m in ["holo_raw", "holo", "fifo", "resv"]:
                v = [r[0][m][0] for r in rs]; mu = np.mean(v); ci = stats.t.ppf(.975, 4) * np.std(v, ddof=1) / np.sqrt(5)
                vr = np.mean([r[0][m][1] for r in rs]); rec[m] = dict(old=float(mu), ci=float(ci), recent=float(vr))
                line += f" {m} {mu:.3f}±{ci:.3f}"
            out[f"b{bits}/D{D}"] = dict(alpha=best, C=C, n_old=[r[2] for r in rs], **rec); print(line, flush=True)
    od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
    json.dump(out, open(os.path.join(od, "e28_kg_gate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
