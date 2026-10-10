"""Six-panel analysis dashboard for a simulation run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from . import analytics
from .book import SimulationResult
from .hawkes import ASK, BID, HawkesParams, intensity_on_grid

THEME = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "border": "#30363d",
    "bid": "#3fb950",
    "ask": "#f85149",
    "mid": "#58a6ff",
    "text": "#e6edf3",
    "muted": "#8b949e",
}


def style_axis(ax, title: str | None = None) -> None:
    C = THEME
    ax.set_facecolor(C["panel"])
    for spine in ax.spines.values():
        spine.set_color(C["border"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    ax.xaxis.label.set_color(C["muted"])
    ax.yaxis.label.set_color(C["muted"])
    if title:
        ax.set_title(title, color=C["text"], fontsize=9, pad=6)


def _legend(ax) -> None:
    ax.legend(fontsize=7, facecolor=THEME["panel"], labelcolor=THEME["text"], framealpha=0.6)


def plot_dashboard(
    res: SimulationResult,
    params: HawkesParams,
    save: str | Path | None = None,
    show: bool = False,
):
    """Draw mid-price, intensities, OFI, inter-arrivals, ACF and summary stats.

    Returns the matplotlib figure. ``save`` writes it to disk; ``show`` opens a
    window (off by default so the function is usable in scripts and tests).
    """
    C = THEME
    T = res.T
    fig = plt.figure(figsize=(16, 10), facecolor=C["bg"])
    fig.suptitle(
        f"Hawkes LOB Simulator  |  μ={params.mu}  α={params.alpha}  "
        f"β={params.beta}  α_x={params.alpha_x}",
        color=C["text"],
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(
        3, 3, figure=fig, hspace=0.45, wspace=0.35, left=0.07, right=0.97, top=0.93, bottom=0.07
    )

    ev_t, ev_s = res.event_times, res.event_sides
    is_bid, is_ask = ev_s == BID, ev_s == ASK

    # 1. Mid-price ----------------------------------------------------------
    a1 = fig.add_subplot(gs[0, :])
    style_axis(a1, "Mid-price & market orders")
    a1.step(res.times, res.mid_prices, where="post", color=C["mid"], lw=0.9, label="Mid-price")
    mids_after = res.mid_prices[1:]
    a1.scatter(
        ev_t[is_bid], mids_after[is_bid], color=C["bid"], s=12, alpha=0.5, label="Buy MO", zorder=3
    )
    a1.scatter(
        ev_t[is_ask], mids_after[is_ask], color=C["ask"], s=12, alpha=0.5, label="Sell MO", zorder=3
    )
    a1.set_xlim(0, T)
    a1.set_ylabel("Price", fontsize=8)
    a1.set_xlabel("Time (s)", fontsize=8)
    _legend(a1)

    # 2. Intensities (exact, on a fine grid) --------------------------------
    a2 = fig.add_subplot(gs[1, :2])
    style_axis(a2, "Conditional intensities λ(t)")
    grid = np.linspace(0.0, T, 3000)
    lam = intensity_on_grid(res.path, params, grid)
    a2.plot(grid, lam[:, BID], color=C["bid"], lw=0.8, label="λ_bid(t)")
    a2.plot(grid, lam[:, ASK], color=C["ask"], lw=0.8, alpha=0.8, label="λ_ask(t)")
    a2.axhline(params.mu, color=C["muted"], lw=0.6, ls="--", label="μ baseline")
    a2.set_xlim(0, T)
    a2.set_ylabel("Intensity (ev/s)", fontsize=8)
    a2.set_xlabel("Time (s)", fontsize=8)
    _legend(a2)

    # 3. Order-flow imbalance -------------------------------------------------
    a3 = fig.add_subplot(gs[1, 2])
    style_axis(a3, "Order flow imbalance (5 s window)")
    ofi_t, ofi = analytics.order_flow_imbalance(ev_t, ev_s, T, window=5.0, step=0.5)
    a3.bar(ofi_t, ofi, width=0.45, color=np.where(ofi >= 0, C["bid"], C["ask"]), alpha=0.8)
    a3.axhline(0, color=C["border"], lw=0.5)
    a3.set_ylim(-1, 1)
    a3.set_ylabel("OFI", fontsize=8)
    a3.set_xlabel("Time (s)", fontsize=8)

    # 4. Inter-arrival distribution ------------------------------------------
    a4 = fig.add_subplot(gs[2, 0])
    style_axis(a4, "Inter-arrival times")
    iat_bid = analytics.inter_arrival_times(ev_t[is_bid])
    iat_ask = analytics.inter_arrival_times(ev_t[is_ask])
    both = np.concatenate([iat_bid, iat_ask])
    if both.size:
        bins = np.linspace(0, np.percentile(both, 95), 30)
        a4.hist(iat_bid, bins=bins, color=C["bid"], alpha=0.6, label="bid", density=True)
        a4.hist(iat_ask, bins=bins, color=C["ask"], alpha=0.4, label="ask", density=True)
        rate = max(is_bid.sum(), 1) / T
        x = np.linspace(0, bins[-1], 100)
        a4.plot(
            x, rate * np.exp(-rate * x), color=C["muted"], lw=1, ls="--", label="Poisson, same rate"
        )
        _legend(a4)
    a4.set_xlabel("Inter-arrival (s)", fontsize=8)
    a4.set_ylabel("Density", fontsize=8)

    # 5. Autocorrelation of buy-order counts ---------------------------------
    a5 = fig.add_subplot(gs[2, 1])
    style_axis(a5, "ACF of buy-order counts (0.2 s bins)")
    lags, acf = analytics.count_autocorrelation(ev_t[is_bid], T, dt=0.2, max_lag=4.0)
    a5.plot(lags, acf, color=C["bid"], lw=1)
    a5.fill_between(lags, acf, 0, alpha=0.12, color=C["bid"])
    a5.axhline(0, color=C["border"], lw=0.5)
    a5.set_xlabel("Lag (s)", fontsize=8)
    a5.set_ylabel("ACF", fontsize=8)

    # 6. Summary --------------------------------------------------------------
    a6 = fig.add_subplot(gs[2, 2])
    style_axis(a6, "Statistics")
    a6.axis("off")
    s = analytics.summary_stats(res, params)
    n = max(s["n_events"], 1)
    rows = [
        ("Total events", f"{s['n_events']}"),
        ("Buy MO", f"{s['n_buy']}  ({100 * s['n_buy'] / n:.0f}%)"),
        ("Sell MO", f"{s['n_sell']}  ({100 * s['n_sell'] / n:.0f}%)"),
        ("Rate (ev/s)", f"{s['rate']:.2f}"),
        ("Stationary rate (ev/s)", f"{s['theoretical_rate']:.2f}"),
        ("Mean λ_bid / λ_ask", f"{s['lambda_bid_mean']:.2f} / {s['lambda_ask_mean']:.2f}"),
        ("Price drift", f"{s['drift']:+.4f}"),
        ("Realised volatility", f"{s['realized_vol']:.4f}"),
        ("Branching (α+α_x)/β", f"{s['branching_ratio']:.3f}"),
        ("Stationary", "yes" if s["stationary"] else "NO (explosive)"),
    ]
    y = 0.97
    for label, val in rows:
        a6.text(0.02, y, label, transform=a6.transAxes, fontsize=7.5, color=C["muted"])
        a6.text(
            0.98,
            y,
            val,
            transform=a6.transAxes,
            fontsize=7.5,
            color=C["text"],
            ha="right",
            fontweight="bold",
        )
        y -= 0.095

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    if show:
        plt.show()
    return fig
