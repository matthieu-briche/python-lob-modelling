"""Pure statistics on simulated order flow (no plotting)."""

from __future__ import annotations

import numpy as np

from .book import SimulationResult
from .hawkes import ASK, BID, HawkesParams, intensity_on_grid


def order_flow_imbalance(
    times, sides, T: float, window: float = 5.0, step: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling ``(N_buy - N_sell) / max(N_buy + N_sell, 1)`` over ``[t - window, t)``.

    Returns ``(grid, ofi)`` with ``grid = window, window + step, … < T``.
    """
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be > 0")
    times = np.asarray(times, dtype=float)
    sides = np.asarray(sides, dtype=int)
    grid = np.arange(window, T, step)

    def counts(side: int) -> np.ndarray:
        ts = times[sides == side]
        return np.searchsorted(ts, grid, "left") - np.searchsorted(ts, grid - window, "left")

    n_bid, n_ask = counts(BID), counts(ASK)
    ofi = (n_bid - n_ask) / np.maximum(n_bid + n_ask, 1)
    return grid, ofi


def inter_arrival_times(times) -> np.ndarray:
    """Waiting times between consecutive events (empty if fewer than two)."""
    times = np.asarray(times, dtype=float)
    return np.diff(times) if times.size > 1 else np.empty(0)


def count_autocorrelation(
    times, T: float, dt: float = 0.2, max_lag: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation of event counts binned at ``dt``.

    Uses a zero-padded FFT so the estimate is linear (not circular). Returns
    ``(lags, acf)`` with ``acf[0] == 1`` whenever counts vary.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    edges = np.arange(0.0, T + dt, dt)
    counts, _ = np.histogram(np.asarray(times, dtype=float), bins=edges)
    x = counts - counts.mean()
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n]
    acf = acov / acov[0] if acov[0] > 0 else np.zeros(n)
    k = min(int(round(max_lag / dt)) + 1, n)
    return np.arange(k) * dt, acf[:k]


def realized_volatility(prices) -> float:
    """Square root of the sum of squared price increments."""
    prices = np.asarray(prices, dtype=float)
    return float(np.sqrt(np.sum(np.diff(prices) ** 2)))


def summary_stats(res: SimulationResult, params: HawkesParams) -> dict[str, float]:
    """Headline numbers for a simulation run."""
    n = len(res.event_times)
    n_bid = int(np.sum(res.event_sides == BID))
    # Time average of λ(t) on a fine grid (averaging at event times is biased
    # upwards, since events cluster where the intensity is high).
    lam = intensity_on_grid(res.path, params, np.linspace(0.0, res.T, 4000))
    return {
        "n_events": n,
        "n_buy": n_bid,
        "n_sell": n - n_bid,
        "rate": n / res.T,
        "theoretical_rate": 2 * params.stationary_rate,
        "lambda_bid_mean": float(lam[:, BID].mean()),
        "lambda_ask_mean": float(lam[:, ASK].mean()),
        "drift": float(res.mid_prices[-1] - res.mid_prices[0]),
        "realized_vol": realized_volatility(res.mid_prices),
        "branching_ratio": params.branching_ratio,
        "stationary": params.is_stationary,
    }
