"""#36 step 4b: conditional normalizing-flow proposal (categorical over 144 material classes x RealNVP over initial state),
exact density, defensive mixture with prior, used as PF proposal. See PREREG.md (4b)."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PF_NTH", "16")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import pf36_g3 as g  # noqa: E402

torch.set_num_threads(16)
HERE = Path(__file__).resolve().parent
ALPHA = .1
ZD = [1, 2, 4, 5, 6]
CEN, HW = (g.HI+g.LO)/2, (g.HI-g.LO)/2
LOGV = float(np.sum(np.log(g.HI-g.LO))) + np.log(144)
LOG_HW = float(np.log(HW).sum())


def features(obs, mask):
    m = np.repeat(mask[:, :g.CUT, None], 2, -1)
    return np.concatenate([(obs[:, :g.CUT]*m).reshape(len(obs), -1), mask[:, :g.CUT]], 1).astype(np.float32)


class Coupling(nn.Module):
    def __init__(self, keep, ctx):
        super().__init__()
        self.register_buffer("keep", torch.tensor(keep, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(5+ctx, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 10))

    def st(self, u, c):
        h = self.net(torch.cat([u*self.keep, c], -1))
        s, t = h[..., :5], h[..., 5:]
        return 2*torch.tanh(s/2)*(1-self.keep), t*(1-self.keep)

    def forward(self, u, c):  # data -> base
        s, t = self.st(u, c)
        return (u-t)*torch.exp(-s), -s.sum(-1)

    def inverse(self, v, c):  # base -> data
        s, t = self.st(v, c)
        return v*torch.exp(s)+t


class Proposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(75, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU())
        self.cat = nn.Linear(128, 144)
        masks = [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]*3
        self.layers = nn.ModuleList([Coupling(m, 128+24) for m in masks])

    def ctx(self, h, cls):
        oh = torch.cat([nn.functional.one_hot(cls//12, 12), nn.functional.one_hot(cls % 12, 12)], -1).float()
        return torch.cat([h, oh], -1)

    def log_prob(self, x, cls, z):
        """x (B,75), cls (B,), z (B,5) raw -> exact log q(z, cls | x)."""
        h = self.enc(x)
        lc = torch.log_softmax(self.cat(h), -1).gather(-1, cls[:, None])[:, 0]
        u = (z-torch.tensor(CEN, dtype=z.dtype))/torch.tensor(HW, dtype=z.dtype)
        c = self.ctx(h, cls)
        ld = 0.
        for layer in self.layers:
            u, l = layer(u, c)
            ld = ld + l
        base = -.5*(u**2).sum(-1) - 2.5*np.log(2*np.pi)
        return lc + base + ld - LOG_HW

    @torch.no_grad()
    def sample(self, x, n, gen):
        """x (C,75) -> cls (C,n), z (C,n,5)."""
        h = self.enc(x)
        probs = torch.softmax(self.cat(h), -1)
        cls = torch.multinomial(probs, n, replacement=True, generator=gen)
        C = len(x)
        hh = h[:, None].expand(C, n, -1).reshape(C*n, -1)
        c = self.ctx(hh, cls.reshape(-1))
        v = torch.randn(C*n, 5, generator=gen, dtype=torch.float32)
        for layer in reversed(self.layers):
            v = layer.inverse(v, c)
        z = v*torch.tensor(HW, dtype=torch.float32)+torch.tensor(CEN, dtype=torch.float32)
        return cls, z.reshape(C, n, 5)


def train(lib, M, seed, log):
    pos, _, st0, im0, id0 = lib
    pos, z, cls = pos[:M], st0[:M][:, ZD].astype(np.float32), (im0[:M]*12+id0[:M]).astype(np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    nv = M//10
    vi, ti = perm[:nv], perm[nv:]
    torch.manual_seed(seed)
    model = Proposal()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def batch(idx, r):
        obs = pos[idx] + r.normal(0, g.SIG, pos[idx].shape)
        mask = np.zeros((len(idx), g.T))
        mask[:, :g.CUT] = r.uniform(size=(len(idx), g.CUT)) > .4
        return torch.tensor(features(obs, mask)), torch.tensor(cls[idx]), torch.tensor(z[idx])
    vx, vc, vz = batch(vi, np.random.default_rng(seed+1))
    best, best_state, bad = np.inf, None, 0
    for ep in range(60):
        model.train()
        order = rng.permutation(ti)
        for s in range(0, len(order), 1024):
            x, c, zz = batch(order[s:s+1024], rng)
            loss = -model.log_prob(x, c, zz).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(-model.log_prob(vx, vc, vz).mean())
        if vl < best-1e-3:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 5:
            break
    model.load_state_dict(best_state)
    log(f"trained M={M} epochs={ep+1} val_nll={best:.3f}")
    return model, dict(epochs=ep+1, val_nll=best)


def proposal_init(model, case, N, seed):
    C = len(case["obs"])
    x = torch.tensor(features(case["obs"], case["mask"]))

    def draw(rng):
        gen = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
        cls, z = model.sample(x, N, gen)
        cls, z = cls.numpy(), z.numpy().astype(float)
        pri = rng.uniform(size=(C, N)) < ALPHA
        z = np.where(pri[..., None], g.LO+(g.HI-g.LO)*rng.uniform(size=(C, N, 5)), z)
        cls = np.where(pri, rng.integers(0, 144, (C, N)), cls)
        with torch.no_grad():
            lq = model.log_prob(x[:, None].expand(C, N, 75).reshape(-1, 75), torch.tensor(cls.reshape(-1)),
                                torch.tensor(z.reshape(-1, 5), dtype=torch.float32)).numpy().reshape(C, N).astype(float)
        inside = np.all((z >= g.LO) & (z <= g.HI), -1)
        lp = np.where(inside, -LOGV, -np.inf)
        lmix = np.logaddexp(np.log(1-ALPHA)+lq, np.log(ALPHA)+lp)
        st = np.zeros((C*N, 7))
        st[:, ZD] = z.reshape(-1, 5)
        return st, (cls//12).ravel(), (cls % 12).ravel(), lp-lmix
    return draw


def summarize(r):
    return {k: float(np.mean(v)) for k, v in r.items()}


def main():
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
    val = g.make_cases(51000, 48, test=False)
    tests = [g.make_cases(s, 48, test=True) for s in range(51200, 51204)]
    libs = {"train_range": g.library(g.TRPOOL, 8, 8, 196_608, 52001), "broad": g.library(g.POOL, 12, 12, 196_608, 52003)}
    xthr = float(np.median(libs["train_range"][0][:, -1, 0]-libs["train_range"][0][:, g.CUT-1, 0]))
    # train-range library indices are into TRPOOL (8x8); map to the PF pool (12x12) nearest classes
    tr = libs["train_range"]
    mi = np.abs(np.log(g.PF_MU)[None]-np.log(g.TR_MU)[:, None]).argmin(1)[tr[3]]
    di = np.abs(g.PF_D[None]-g.TR_D[:, None]).argmin(1)[tr[4]]
    libs["train_range"] = (tr[0], tr[1], tr[2], mi, di)
    log("libs ready")
    a4 = json.loads((HERE/"gate4a.json").read_text(encoding="utf-8"))
    ret_key = a4["best_retrieval"]
    ret, ret_seeds = a4["table"][ret_key]["energy"], np.array(a4["table"][ret_key]["energy_per_seed"])
    a4_best = min(v["energy"] for k, v in a4["table"].items() if k.startswith("rpf_") and v["within_budget"])
    ref = a4["reference"]["pf_N8192"]["energy"]
    table, verdict = {}, {}
    for variant in ("train_range", "broad"):
        for M, N in ((98_304, 512), (147_456, 256)):
            model, info = train(libs[variant], M, 61000+M//1000, log)
            vt = {tp: float(np.mean(g.run_pf_full(val, N, tp, xthr, 3, init=proposal_init(model, val, N, 0),
                                                 resample_after=g.CUT-1)["energy"])) for tp in (1, 2, 4)}
            tp = min(vt, key=vt.get)
            per = [summarize(g.run_pf_full(c, N, tp, xthr, 500+N+i, init=proposal_init(model, c, N, i), resample_after=g.CUT-1))
                   for i, c in enumerate(tests)]
            key = f"flowpf_{variant}_M{M}_N{N}"
            e = float(np.mean([p["energy"] for p in per]))
            es = np.array([p["energy"] for p in per])
            table[key] = dict({k: float(np.mean([p[k] for p in per])) for k in per[0]}, energy_per_seed=es.tolist(),
                              train_sims=M, calls_per_case=N, calls_total=M+N*192, pf_temp=tp, val=vt, train=info)
            verdict[key] = dict(vs_ref=e/ref-1, vs_retrieval=e/ret-1, vs_4a_best=e/a4_best-1,
                                all_seeds_beat_retrieval=bool(np.all(es < ret_seeds)),
                                passed=bool(M+N*192 <= 196_608 and e <= 1.05*ref and e <= .95*ret and np.all(es < ret_seeds) and e < a4_best))
            log(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in table[key].items() if not isinstance(v, (list, dict))}, verdict[key])
    out = dict(reference=dict(pf_N8192=ref, best_retrieval=ret_key, best_retrieval_energy=ret, best_4a_in_budget=a4_best),
               table=table, verdict=verdict, passed=any(v["passed"] for v in verdict.values()), seconds=time.time()-t0)
    (HERE/"gate4b.json").write_text(json.dumps(out, indent=1)+"\n", encoding="utf-8")
    log("done passed", out["passed"])


if __name__ == "__main__":
    main()
