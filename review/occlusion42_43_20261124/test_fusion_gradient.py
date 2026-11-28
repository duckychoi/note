"""Regression for the adaptive gain reaching the student through its auxiliary loss."""
import torch

from depth_full_train import fusion_supervision


def test_gain_gradient():
    # One valid pixel makes the analytic gradient independent of any batch scaling.
    z = torch.tensor([0., 0.], requires_grad=True)
    neighbor = torch.tensor([2., 2.], requires_grad=True)
    shift = torch.tensor([0., 0.], requires_grad=True)
    valid = torch.tensor([True, False])
    for k in (0., .1, .5, .9, 1.):
        gain = torch.full_like(z, k, requires_grad=True)
        for arm, expected in (("kf", -2*k), ("fixed", -1.)):
            value, _ = fusion_supervision(z, neighbor, shift, gain, valid, arm)
            grads = torch.autograd.grad(value, (z, neighbor, shift, gain), allow_unused=True)
            torch.testing.assert_close(grads[0], torch.tensor([expected, 0.]))
            assert grads[1:] == (None, None, None)
    empty, _ = fusion_supervision(z, neighbor, shift, gain, ~torch.ones_like(valid), "kf")
    torch.testing.assert_close(torch.autograd.grad(empty, z)[0], torch.zeros_like(z))


if __name__ == "__main__":
    test_gain_gradient()
    print("Fixed and adaptive gain gradients, detached targets and empty-mask behavior passed.")
