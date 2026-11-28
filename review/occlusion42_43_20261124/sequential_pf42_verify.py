"""CPU verification of the exact transition gradient and finite-particle evidence."""
import itertools
from pathlib import Path
import json

import numpy as np
import torch


OUT = Path(__file__).resolve().parents[3] / "research-sources/encoder42_eval/sequential-pf42"


def main():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    logits = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    prior = torch.tensor([.25, .5, .25], dtype=torch.float64)
    states = torch.arange(3, dtype=torch.float64)
    transition = (logits[:, None] - (states[:, None] - states)**2).softmax(-1)
    likelihood = torch.rand(4, 3, dtype=torch.float64).mul(.9).add(.05)
    vector = prior * likelihood[0]
    for kernel, measure in zip(transition, likelihood[1:]):
        vector = (vector @ kernel) * measure
    value = vector.sum()
    grad, = torch.autograd.grad(-value.log(), logits)
    path = torch.tensor(list(itertools.product(range(3), repeat=4)))
    weight = prior[path[:, 0]] * likelihood[0, path[:, 0]]
    for t in range(1, 4):
        weight = weight * transition[t - 1, path[:, t - 1], path[:, t]] * likelihood[t, path[:, t]]
    torch.testing.assert_close(weight.sum().detach(), value.detach())
    path_mass = weight.detach() / weight.sum().detach()
    exact = []
    for t, kernel in enumerate(transition.detach()):
        last = torch.bincount(path[:, t], weights=path_mass, minlength=3)
        now = torch.bincount(path[:, t + 1], weights=path_mass, minlength=3)
        exact.append(last @ kernel - now)
    gradient_error = float((grad - torch.stack(exact)).abs().max())
    assert gradient_error < 1e-12
    runs, particles = 20000, 64
    live = np.broadcast_to((prior * likelihood[0]).numpy(), (runs, 3)).copy()
    for kernel, measure in zip(transition.detach().numpy(), likelihood[1:].numpy()):
        mass = live.sum(1)
        cumulative = (live / mass[:, None]).cumsum(1)[:, :-1]
        picked = (rng.random((runs, particles, 1)) > cumulative[:, None]).sum(2)
        predicted = kernel[picked].mean(1)
        live = mass[:, None] * predicted * measure[None]
    evidence = live.sum(1)
    precise = float(value.detach())
    error = evidence.std(ddof=1) / np.sqrt(runs)
    assert abs(evidence.mean() - precise) < 5 * error
    assert np.log(evidence).mean() <= np.log(precise) + 5 * np.log(evidence).std(ddof=1) / np.sqrt(runs)
    result = dict(gradient_max_abs_error=gradient_error, exact_joint=precise, particles=particles, trials=runs,
                  particle_joint_mean=float(evidence.mean()), standard_error=float(error),
                  exact_log=float(np.log(precise)), mean_particle_log=float(np.log(evidence).mean()))
    (OUT / "gradient-check.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(result)


if __name__ == "__main__":
    main()
