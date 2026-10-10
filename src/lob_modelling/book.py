"""Stylised limit order book driven by Hawkes market-order flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hawkes import BID, HawkesPath, HawkesProcess


@dataclass
class SimulationResult:
    """Output of :meth:`LimitOrderBook.run`.

    ``times``, ``mid_prices``, ``lambda_bid`` and ``lambda_ask`` start with the
    initial state at ``t = 0`` followed by one entry per event, so they have
    ``len(path) + 1`` elements. Intensities are the left limits ``λ(t-)``.
    """

    path: HawkesPath
    times: np.ndarray
    mid_prices: np.ndarray
    lambda_bid: np.ndarray
    lambda_ask: np.ndarray

    @property
    def event_times(self) -> np.ndarray:
        return self.path.times

    @property
    def event_sides(self) -> np.ndarray:
        return self.path.sides

    @property
    def T(self) -> float:
        return self.path.T


class LimitOrderBook:
    """Book whose mid-price moves with each market order.

    A buy market order (bid process) lifts the mid-price by a random fraction
    of a tick drawn uniformly in ``impact``; a sell order lowers it. The book
    keeps a constant two-tick spread and random queue sizes for display.
    """

    def __init__(
        self,
        mid_price: float = 100.0,
        tick_size: float = 0.01,
        depth: int = 5,
        mu: float = 2.0,
        alpha: float = 0.6,
        beta: float = 1.5,
        alpha_x: float = 0.2,
        impact: tuple[float, float] = (0.2, 0.8),
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        lo, hi = impact
        if not 0 <= lo <= hi:
            raise ValueError("impact must satisfy 0 <= low <= high")
        self.rng = np.random.default_rng(rng)
        self.mid0 = float(mid_price)
        self.mid_price = float(mid_price)
        self.tick = float(tick_size)
        self.depth = int(depth)
        self.impact = (float(lo), float(hi))
        self.hawkes = HawkesProcess(mu, alpha, beta, alpha_x, rng=self.rng)
        self.bids: list[tuple[float, int]] = []
        self.asks: list[tuple[float, int]] = []
        self._queues = self._draw_queues()
        self.refresh_levels()

    @property
    def spread(self) -> float:
        return 2 * self.tick

    # Book levels ------------------------------------------------------------
    def _draw_queues(self) -> np.ndarray:
        base = (self.depth - np.arange(self.depth)) * 8
        return self.rng.integers(10, 100, size=(2, self.depth)) + base

    def refresh_levels(self, churn: float = 0.0) -> None:
        """Rebuild price levels around the mid; ``churn`` ∈ [0, 1] resamples queues."""
        if churn > 0:
            fresh = self._draw_queues()
            mask = self.rng.random(self._queues.shape) < churn
            self._queues = np.where(mask, fresh, self._queues)
        best_bid = self.mid_price - self.spread / 2
        best_ask = self.mid_price + self.spread / 2
        self.bids = [
            (round(best_bid - i * self.tick, 6), int(self._queues[0, i])) for i in range(self.depth)
        ]
        self.asks = [
            (round(best_ask + i * self.tick, 6), int(self._queues[1, i])) for i in range(self.depth)
        ]

    # Dynamics ---------------------------------------------------------------
    def apply_market_order(self, side: int) -> float:
        """Move the mid-price for one market order and return the new mid."""
        move = self.tick * self.rng.uniform(*self.impact)
        self.mid_price += move if side == BID else -move
        return self.mid_price

    def run(self, T: float = 60.0) -> SimulationResult:
        """Simulate ``T`` seconds of order flow from the initial mid-price."""
        path = self.hawkes.simulate(T)
        self.mid_price = self.mid0
        mids = np.empty(len(path) + 1)
        mids[0] = self.mid0
        for k, side in enumerate(path.sides, start=1):
            mids[k] = self.apply_market_order(int(side))
        self.refresh_levels()

        mu = self.hawkes.mu
        return SimulationResult(
            path=path,
            times=np.concatenate([[0.0], path.times]),
            mid_prices=mids,
            lambda_bid=np.concatenate([[mu], path.intensity_before[:, 0]]),
            lambda_ask=np.concatenate([[mu], path.intensity_before[:, 1]]),
        )
