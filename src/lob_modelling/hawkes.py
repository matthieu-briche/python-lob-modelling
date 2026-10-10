"""Bivariate Hawkes process with an exponential kernel.

Intensity of process ``i`` (0 = bid / buy, 1 = ask / sell)::

    λ_i(t) = μ + Σ_j Σ_{t_k^j < t} α_ij · exp(-β (t - t_k^j))

with ``α_ii = alpha`` (self-excitation) and ``α_ij = alpha_x`` for ``i ≠ j``
(cross-excitation). Simulation uses Ogata's (1981) thinning algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BID, ASK = 0, 1


@dataclass(frozen=True)
class HawkesParams:
    """Parameters of the symmetric bivariate Hawkes process."""

    mu: float = 2.0
    alpha: float = 0.6
    beta: float = 1.5
    alpha_x: float = 0.2

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError("mu must be > 0")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
        if self.alpha < 0 or self.alpha_x < 0:
            raise ValueError("alpha and alpha_x must be >= 0")

    @property
    def branching_ratio(self) -> float:
        """Spectral radius of the branching matrix, ``(α + α_x) / β``."""
        return (self.alpha + self.alpha_x) / self.beta

    @property
    def is_stationary(self) -> bool:
        return self.branching_ratio < 1.0

    @property
    def stationary_rate(self) -> float:
        """Long-run event rate of *each* side, ``μ / (1 - (α + α_x)/β)``."""
        if not self.is_stationary:
            return float("inf")
        return self.mu / (1.0 - self.branching_ratio)


@dataclass
class HawkesPath:
    """A simulated trajectory.

    ``intensity_before`` holds ``(λ_bid(t-), λ_ask(t-))`` just before each event.
    """

    times: np.ndarray
    sides: np.ndarray
    intensity_before: np.ndarray
    T: float

    def __len__(self) -> int:
        return len(self.times)

    def side_times(self, side: int) -> np.ndarray:
        return self.times[self.sides == side]


class HawkesProcess:
    """Bivariate Hawkes process simulated event by event.

    The state ``R[i, j]`` is the excitation that source ``j`` currently adds to
    the intensity of process ``i``. With an exponential kernel it simply decays
    by ``exp(-β dt)`` between events, which makes simulation O(1) per event.
    """

    def __init__(
        self,
        mu: float = 2.0,
        alpha: float = 0.6,
        beta: float = 1.5,
        alpha_x: float = 0.2,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        self.params = HawkesParams(mu=mu, alpha=alpha, beta=beta, alpha_x=alpha_x)
        self.rng = np.random.default_rng(rng)
        self.reset()

    # Convenience accessors ------------------------------------------------
    @property
    def mu(self) -> float:
        return self.params.mu

    @property
    def alpha(self) -> float:
        return self.params.alpha

    @property
    def beta(self) -> float:
        return self.params.beta

    @property
    def alpha_x(self) -> float:
        return self.params.alpha_x

    # State ----------------------------------------------------------------
    def reset(self) -> None:
        self.R = np.zeros((2, 2))
        self.t = 0.0
        self.last_intensity_before = (self.mu, self.mu)

    def intensities(self) -> tuple[float, float]:
        """Return ``(λ_bid, λ_ask)`` at the current time."""
        lam = self.mu + self.R.sum(axis=1)
        return float(lam[BID]), float(lam[ASK])

    def _decay(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("cannot decay by a negative time step")
        self.R *= np.exp(-self.beta * dt)

    def _trigger(self, side: int) -> None:
        self.R[side, side] += self.alpha
        self.R[1 - side, side] += self.alpha_x

    # Simulation -----------------------------------------------------------
    def next_event(self, horizon: float = np.inf) -> tuple[float, int] | None:
        """Draw the next event by Ogata thinning.

        Between events the intensity only decreases, so the current total
        intensity is a valid upper bound. Returns ``(t, side)``, or ``None``
        if no event occurs before ``horizon`` (the clock is then set to
        ``horizon`` and no event is recorded).
        """
        while True:
            lam_bar = sum(self.intensities())
            dt = self.rng.exponential(1.0 / lam_bar)
            if self.t + dt > horizon:
                self._decay(horizon - self.t)
                self.t = horizon
                return None
            self._decay(dt)
            self.t += dt

            lb, la = self.intensities()
            u = self.rng.uniform(0.0, lam_bar)
            if u < lb:
                side = BID
            elif u < lb + la:
                side = ASK
            else:
                continue  # rejected candidate
            self.last_intensity_before = (lb, la)
            self._trigger(side)
            return self.t, side

    def simulate(self, T: float) -> HawkesPath:
        """Simulate on ``[0, T]`` from an empty history."""
        if T <= 0:
            raise ValueError("T must be > 0")
        self.reset()
        times: list[float] = []
        sides: list[int] = []
        lam_before: list[tuple[float, float]] = []
        while True:
            ev = self.next_event(horizon=T)
            if ev is None:
                break
            t, side = ev
            times.append(t)
            sides.append(side)
            lam_before.append(self.last_intensity_before)
        return HawkesPath(
            times=np.asarray(times, dtype=float),
            sides=np.asarray(sides, dtype=int),
            intensity_before=np.asarray(lam_before, dtype=float).reshape(-1, 2),
            T=float(T),
        )


def intensity_on_grid(path: HawkesPath, params: HawkesParams, grid: np.ndarray) -> np.ndarray:
    """Exact ``λ(t)`` evaluated on an increasing time grid, shape ``(len(grid), 2)``."""
    grid = np.asarray(grid, dtype=float)
    out = np.empty((len(grid), 2))
    R = np.zeros((2, 2))
    t_prev = 0.0
    k = 0
    n = len(path)
    for g, t in enumerate(grid):
        while k < n and path.times[k] <= t:
            R *= np.exp(-params.beta * (path.times[k] - t_prev))
            t_prev = path.times[k]
            s = path.sides[k]
            R[s, s] += params.alpha
            R[1 - s, s] += params.alpha_x
            k += 1
        out[g] = params.mu + (R * np.exp(-params.beta * (t - t_prev))).sum(axis=1)
    return out
