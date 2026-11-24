"""Change only the material quantiles; keep the trial's initial state, mask and innovations."""
from fresh import HERE, SOURCE, SEEDS, METRICS, flow, g, np, read, torch


def draw(rng, low, high, n):
    left = rng.uniform(size=n) < .5
    a = rng.uniform(*low,n)
    b = rng.uniform(*high,n)
    actual = np.where(left,a,b)
    quantile = .5*np.where(left,(a-low[0])/(low[1]-low[0]),1+(b-high[0])/(high[1]-high[0]))
    return actual, quantile


def matched(seed):
    rng = np.random.default_rng(seed)
    n = 24
    mu, u = draw(rng,(.08,.22),(.9,1.2),n)
    d, v = draw(rng,(.07,.17),(.55,.8),n)
    rng_state = rng.bit_generator.state
    pair = []
    for friction, damp in ((mu,d),(.3+.5*u,.2+.3*v)):
        rng.bit_generator.state = rng_state
        models = [g.mk(a,b) for a,b in zip(friction,damp)]
        pos, bnc = g.simulate(models,g.init_states(rng,n),rng,g.T)
        obs = pos + rng.normal(0,g.SIG,pos.shape)
        mask = np.zeros((n,g.T))
        mask[:,:g.CUT] = rng.uniform(size=(n,g.CUT))>.4
        pair.append(dict(pos=pos,bnc=bnc,obs=obs,mask=mask,mu=friction,d=damp))
    outer, inner = pair
    original = g.make_cases(seed,n,test=True)
    for k in outer:
        np.testing.assert_array_equal(outer[k],original[k])
    np.testing.assert_array_equal(inner["mask"],outer["mask"])
    np.testing.assert_array_equal(inner["pos"][:,0],outer["pos"][:,0])
    np.testing.assert_allclose(inner["obs"]-inner["pos"],outer["obs"]-outer["pos"],atol=1e-14)
    return inner


if __name__ == "__main__":
    torch.set_num_threads(3)
    threshold = read("gate3.json")["config"]["xthr"]
    weights = torch.load(SOURCE/"gate5_primary_models.pt",map_location="cpu",weights_only=True)
    runs = [r for r in read("gate5.json")["runs"] if r["role"]=="primary"]
    models = []
    for r in runs:
        m = flow.Proposal()
        m.load_state_dict(weights[f"{r['M']}_{r['train_seed']}"])
        models.append(m.eval())
    for i, seed in enumerate(SEEDS):
        c = matched(seed)
        raw = {f"case_{k}":v for k,v in c.items()}
        for r,m in zip(runs,models):
            n = r["N"]
            scores = g.run_pf_full(c,n,r["pf_temp"],threshold,66900+i,
                init=flow.proposal_init(m,c,n,0),resample_after=g.CUT-1)
            raw.update({f"flow_{r['train_seed']}_{k}":scores[k] for k in METRICS})
        np.savez_compressed(HERE/f"inner_{seed}.npz",**raw)
