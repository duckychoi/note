"""
#28 audit (11-22): does the FHRR + complex-SSM toy support "creativity = long memory + phase constructive interference"?

A  T2 control      : the 'creative' item in the original T2 is a random new combination that the experimenter WRITES into
                     memory and then reads back by its ID.  Control = read back an ordinary stored item at the same load.
                     If both curves coincide, T2 measures recall capacity, not creativity.
B  real recombination: build an item that was NEVER stored, purely by algebra from two stored items
                     (role-0 value of item a + role-1 value of item b + role-2 value of item c), and read it out.
C  capacity theory : T1 against the textbook crosstalk prediction (Plate 1995; Frady, Kleyko & Sommer 2018):
                     sim to target ~ N(1, s2), to each distractor ~ N(0, s2), s2 = (N-1)/(2D);
                     P(correct) = E_z[ Phi((1 + s z)/s)^(M-1) ].  No duplicate pairs (original N=150 > M=100 reused pairs).
D  long memory cost: complex-SSM stream of L=400 items; recall at lag 5/25/100 for alpha in {0.003,...,0.3}.
                     Is there one best alpha (long memory always better) or a lag-dependent optimum (trade-off)?
Results -> results/v2/e28_audit.json
"""
import os, json
import numpy as np
from scipy.stats import norm

D = 1024; rng = np.random.default_rng(0)
ph = lambda n: np.exp(1j * rng.uniform(-np.pi, np.pi, (n, D)))
sim = lambda a, B: np.real(np.conj(B) @ a) / D
out = {}

# ---------------- A + B
R, V, MAXID, T = 3, 20, 400, 200
RoleV = ph(R); Vocab = [ph(V) for _ in range(R)]; IDs = ph(MAXID)
item = lambda vals: sum(RoleV[r] * Vocab[r][vals[r]] for r in range(R))
read = lambda rec: [int(np.argmax(sim(np.conj(RoleV[r]) * rec, Vocab[r]))) for r in range(R)]
print("A/B  load L | creative(orig T2) | ordinary stored item | never-stored recombination (a.r0 + b.r1 + c.r2)")
out["AB"] = {}
for L in [10, 30, 60, 100, 150]:
    a_ok = c_ok = b_ok = b1_ok = 0
    for _ in range(T):
        ids = rng.choice(MAXID, L, replace=False); vals = [list(rng.integers(V, size=R)) for _ in range(L)]
        mem = sum(IDs[ids[j]] * item(vals[j]) for j in range(L))
        a_ok += read(np.conj(IDs[ids[0]]) * mem) == vals[0]            # orig T2: item 0 = 'creative'
        q = rng.integers(1, L); c_ok += read(np.conj(IDs[ids[q]]) * mem) == vals[q]   # ordinary stored item
        a_, b_, c_ = rng.choice(L, 3, replace=False)                   # never-stored recombination
        # B0 pure superposition (no clean-up): unbind role r of item s, rebind to role r, sum -> role factors cancel
        new = sum(RoleV[r] * (np.conj(RoleV[r]) * np.conj(IDs[ids[s]]) * mem) for r, s in enumerate([a_, b_, c_]))
        b_ok += read(new) == [vals[a_][0], vals[b_][1], vals[c_][2]]
        # B1 with symbolic clean-up of each part (codebook nearest neighbour) before recomposition
        parts = [int(np.argmax(sim(np.conj(RoleV[r]) * np.conj(IDs[ids[s]]) * mem, Vocab[r]))) for r, s in enumerate([a_, b_, c_])]
        new1 = sum(RoleV[r] * Vocab[r][parts[r]] for r in range(R))
        b1_ok += read(new1) == [vals[a_][0], vals[b_][1], vals[c_][2]]
    out["AB"][L] = dict(creative=a_ok / T, ordinary=c_ok / T, recombined_superpos=b_ok / T, recombined_cleanup=b1_ok / T)
    print(f"  {L:>8} | {a_ok/T:17.2f} | {c_ok/T:20.2f} | superpos {b_ok/T:.2f} / clean-up {b1_ok/T:.2f}")

# ---------------- C
print("\nC  N pairs (M=N codebook, no duplicates) | empirical | theory")
z = rng.standard_normal(20000); out["C"] = {}
for N in [50, 100, 150, 200, 300, 400]:
    Ro, Fi = ph(N), ph(N); mem = (Ro * Fi).sum(0); acc = []
    for q in rng.choice(N, 200):
        acc.append(int(np.argmax(sim(np.conj(Ro[q]) * mem, Fi))) == q)
    s = np.sqrt((N - 1) / (2 * D)); th = float(np.mean(norm.cdf((1 + s * z) / s) ** (N - 1)))
    out["C"][N] = dict(emp=float(np.mean(acc)), theory=th); print(f"  {N:>6} | {np.mean(acc):.2f} | {th:.2f}")

# ---------------- D
print("\nD  complex SSM, stream L=400 | recall @ lag 5 / 25 / 100")
L = 400; Times, Items = ph(L), ph(L); out["D"] = {}; lags = [5, 25, 100]
for alpha in [0.003, 0.01, 0.04, 0.12, 0.3]:
    lam = np.exp(-alpha); x = np.zeros(D, complex); rec = {k: [] for k in lags}
    for t in range(L):
        x = lam * x + Times[t] * Items[t]
        if t >= 150:
            for k in lags:
                rec[k].append(int(np.argmax(sim(np.conj(Times[t - k]) * x, Items))) == t - k)
    r = {k: float(np.mean(v)) for k, v in rec.items()}; out["D"][alpha] = r
    print(f"  alpha {alpha:5.3f} (len {1/alpha:6.1f}) | " + " / ".join(f"{r[k]:.2f}" for k in lags))

od = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "v2")
json.dump(out, open(os.path.join(od, "e28_audit.json"), "w"), indent=1)
