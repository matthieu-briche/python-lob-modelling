"""Maximum likelihood calibration of exponential Hawkes processes.

Both log-likelihoods use Ozaki's (1979) O(n) recursion for the excitation at
event times. Stationarity is not imposed: the likelihood on a finite window is
well defined either way, and the fit reports the estimated branching ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-300
_LOWER = 1e-6


@dataclass
class CalibrationResult:
    mu: float
    alpha: float
    beta: float
    log_likelihood: float
    success: bool
    alpha_x: float = 0.0
    message: str = field(default="", repr=False)

    @property
    def branching_ratio(self) -> float:
        return (self.alpha + self.alpha_x) / self.beta


def _check_times(event_times, T: float) -> np.ndarray:
    t = np.asarray(event_times, dtype=float)
    if t.ndim != 1:
        raise ValueError("event_times must be one-dimensional")
    if T <= 0:
        raise ValueError("T must be > 0")
    if t.size and (np.any(np.diff(t) < 0) or t[0] < 0 or t[-1] > T):
        raise ValueError("event_times must be sorted and lie in [0, T]")
    return t


# Univariate --------------------------------------------------------------------
def hawkes_loglik(event_times, T: float, mu: float, alpha: float, beta: float) -> float:
    """Exact log-likelihood of a univariate exponential Hawkes process on ``[0, T]``."""
    t = _check_times(event_times, T)
    n = t.size
    A = np.zeros(n)  # A[i] = Σ_{k<i} exp(-β (t_i - t_k))
    if n > 1:
        decay = np.exp(-beta * np.diff(t))
        for i in range(1, n):
            A[i] = decay[i - 1] * (1.0 + A[i - 1])
    log_term = np.sum(np.log(mu + alpha * A + _EPS))
    compensator = mu * T + alpha / beta * np.sum(1.0 - np.exp(-beta * (T - t)))
    return float(log_term - compensator)


def mle_hawkes(
    event_times,
    T: float,
    mu0: float = 1.0,
    alpha0: float = 0.5,
    beta0: float = 2.0,
) -> CalibrationResult:
    """Fit ``(μ, α, β)`` of a univariate Hawkes process by L-BFGS-B."""
    t = _check_times(event_times, T)
    if t.size < 2:
        raise ValueError("need at least two events to calibrate")

    def nll(x):
        return -hawkes_loglik(t, T, *x)

    res = minimize(
        nll,
        [mu0, alpha0, beta0],
        method="L-BFGS-B",
        bounds=[(_LOWER, None)] * 3,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    mu, alpha, beta = res.x
    return CalibrationResult(
        mu=float(mu),
        alpha=float(alpha),
        beta=float(beta),
        log_likelihood=float(-res.fun),
        success=bool(res.success),
        message=str(res.message),
    )


# Bivariate (symmetric) ---------------------------------------------------------
def bivariate_loglik(
    times, sides, T: float, mu: float, alpha: float, alpha_x: float, beta: float
) -> float:
    """Log-likelihood of the symmetric bivariate model used by the simulator.

    ``times`` is the merged, sorted event stream and ``sides`` ∈ {0, 1} labels
    each event.
    """
    t = _check_times(times, T)
    s = np.asarray(sides, dtype=int)
    if s.shape != t.shape:
        raise ValueError("times and sides must have the same length")
    S = np.zeros(2)  # S[j] = Σ_{t_k^j < t} exp(-β (t - t_k^j))
    t_prev = 0.0
    log_term = 0.0
    for ti, si in zip(t, s, strict=True):
        S *= np.exp(-beta * (ti - t_prev))
        t_prev = ti
        lam = mu + alpha * S[si] + alpha_x * S[1 - si]
        log_term += np.log(lam + _EPS)
        S[si] += 1.0
    tail = 1.0 - np.exp(-beta * (T - t))  # each event's integrated kernel
    # Each event excites its own side with α and the other side with α_x.
    compensator = 2 * mu * T + (alpha + alpha_x) / beta * tail.sum()
    return float(log_term - compensator)


def mle_bivariate(
    times,
    sides,
    T: float,
    mu0: float = 1.0,
    alpha0: float = 0.4,
    alpha_x0: float = 0.1,
    beta0: float = 1.0,
) -> CalibrationResult:
    """Fit ``(μ, α, α_x, β)`` of the symmetric bivariate Hawkes model."""
    t = _check_times(times, T)
    if t.size < 2:
        raise ValueError("need at least two events to calibrate")
    s = np.asarray(sides, dtype=int)

    def nll(x):
        mu, alpha, alpha_x, beta = x
        return -bivariate_loglik(t, s, T, mu, alpha, alpha_x, beta)

    res = minimize(
        nll,
        [mu0, alpha0, alpha_x0, beta0],
        method="L-BFGS-B",
        bounds=[(_LOWER, None), (0.0, None), (0.0, None), (_LOWER, None)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    mu, alpha, alpha_x, beta = res.x
    return CalibrationResult(
        mu=float(mu),
        alpha=float(alpha),
        alpha_x=float(alpha_x),
        beta=float(beta),
        log_likelihood=float(-res.fun),
        success=bool(res.success),
        message=str(res.message),
    )
