"""
Exact flow-matching quantities for isotropic Gaussian mixtures (any dim d).

Interpolant (same as the notes): x_t = (1-t) x0 + t x1, x0 ~ N(0, I), x1 ~ p (GMM).
Component k: x1 ~ N(mu_k, s_k^2 I)  =>  x_t | k ~ N(t mu_k, V_k(t) I),  V_k = t^2 s_k^2 + (1-t)^2.

Closed forms used everywhere (no kernel estimator, no NN):
  m(x,t)  = E[x1 | x_t = x] = sum_k g_k(x,t) [ mu_k + t s_k^2 / V_k (x - t mu_k) ]
  u(x,t)  = E[x1 - x0 | x_t = x] = (m - x) / (1 - t)          (marginal FM velocity)
  s(x,t)  = grad log p_t(x)       = -(x - t m) / (1 - t)^2      (score)
"""
import numpy as np


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class GMM:
    def __init__(self, w, mu, s):
        w = np.asarray(w, float)
        if w.ndim != 1 or len(w) == 0 or np.any(~np.isfinite(w)) or np.any(w < 0) or not (0 < w.sum() < np.inf):
            raise ValueError(f"weights must be finite, non-negative, non-empty with positive sum; got {w}")
        self.w = w / w.sum()
        mu = np.asarray(mu, float)
        if mu.ndim > 2 or np.any(~np.isfinite(mu)):
            raise ValueError(f"means must be finite with ndim <= 2; got shape {mu.shape}")
        self.mu = np.atleast_2d(mu)                                # (K,d)
        if self.mu.shape[0] != len(self.w):
            self.mu = self.mu.T
        if self.mu.shape[0] != len(self.w):
            raise ValueError(f"{len(self.w)} weights but means have shape {self.mu.shape}")
        self.s = np.broadcast_to(np.asarray(s, float), self.w.shape).copy()
        if np.any(~np.isfinite(self.s)) or np.any(self.s <= 0) or np.any(self.s < 1e-140) or np.any(self.s > 1e150):
            # supported (numerically verified) range: s in [1e-140, 1e150] and |x|, |mu| <= 1e100 (squared distances stay finite)
            raise ValueError(f"component std must be finite and > 0; got {self.s}")
        self.d = self.mu.shape[1]

    def _X(self, X):
        X = np.asarray(X, float)
        if X.ndim not in (1, 2):
            raise ValueError(f"expected (n,d) or (n,) input, got shape {X.shape}")
        if X.ndim == 1:
            # 1-D model: a vector means n points; d-dim model: a vector means one point
            X = X[:, None] if self.d == 1 else X[None, :]
        if X.shape[-1] != self.d:
            raise ValueError(f"last dim {X.shape[-1]} != d={self.d}")
        return X

    # ---- marginal of the interpolant at time t (t=1 -> data) ----
    def _comp_logpdf_t(self, X, t):
        V = t**2 * self.s**2 + (1 - t)**2                       # (K,)
        d2 = ((X[:, None, :] - t * self.mu[None]) ** 2).sum(-1)  # (n,K)
        return -0.5 * d2 / V - 0.5 * self.d * np.log(2 * np.pi * V) + np.log(self.w), V

    def _resp(self, X, t):
        e, V = self._comp_logpdf_t(X, t)
        return np.exp(e - _lse(e, 1)[:, None]), V

    def logpdf_t(self, X, t):
        X = self._X(X)
        e, _ = self._comp_logpdf_t(X, t)
        return _lse(e, 1)

    def logpdf(self, X):
        return self.logpdf_t(X, 1.0)

    def post_mean(self, X, t):
        """m(x,t) = E[x1 | x_t = x]."""
        X = self._X(X)
        g, V = self._resp(X, t)
        gain = t * self.s**2 / V
        mk = self.mu[None] + gain[None, :, None] * (X[:, None, :] - t * self.mu[None])
        return (g[..., None] * mk).sum(1)

    def velocity(self, X, t):
        """Component-wise stable form: u_k = mu_k + (t s_k^2 - (1-t)) (x - t mu_k) / V_k (valid at t->1)."""
        X = self._X(X)
        g, V = self._resp(X, t)
        uk = self.mu[None] + ((t * self.s**2 - (1 - t)) / V)[None, :, None] * (X[:, None, :] - t * self.mu[None])
        return (g[..., None] * uk).sum(1)

    def score(self, X, t):
        X = self._X(X)
        g, V = self._resp(X, t)
        sk = -(X[:, None, :] - t * self.mu[None]) / V[None, :, None]
        return (g[..., None] * sk).sum(1)

    def sample(self, n, rng):
        k = rng.choice(len(self.w), size=n, p=self.w)
        return self.mu[k] + self.s[k][:, None] * rng.standard_normal((n, self.d))

    def sample_t(self, n, t, rng):
        x1 = self.sample(n, rng)
        return (1 - t) * rng.standard_normal((n, self.d)) + t * x1


def gl(a, b, n):
    """Gauss-Legendre nodes/weights on [a,b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def gh(n):
    """Probabilists' Gauss-Hermite: E_{z~N(0,1)} f(z) ~= sum w f(z)."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / np.sqrt(2 * np.pi)
