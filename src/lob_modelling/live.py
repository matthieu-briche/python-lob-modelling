"""Real-time matplotlib animation of the book."""

from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation

from .book import LimitOrderBook
from .plotting import THEME, style_axis


class LiveSimulation:
    """Rolling state for the animation, independent of any drawing code."""

    def __init__(self, book: LimitOrderBook, history: int = 500) -> None:
        self.book = book
        self.book.hawkes.reset()
        self.times: deque[float] = deque(maxlen=history)
        self.prices: deque[float] = deque(maxlen=history)
        self.lambda_bid: deque[float] = deque(maxlen=history)
        self.lambda_ask: deque[float] = deque(maxlen=history)

    @property
    def t(self) -> float:
        return self.book.hawkes.t

    def step(self, n_events: int = 3) -> None:
        """Advance by ``n_events`` market orders and record the new state."""
        hp = self.book.hawkes
        for _ in range(n_events):
            _, side = hp.next_event()
            self.book.apply_market_order(side)
            lb, la = hp.intensities()
            self.times.append(hp.t)
            self.prices.append(self.book.mid_price)
            self.lambda_bid.append(lb)
            self.lambda_ask.append(la)
        self.book.refresh_levels(churn=0.15)


def _draw_book(ax, book: LimitOrderBook) -> None:
    C = THEME
    ax.cla()
    style_axis(ax, "LOB snapshot")
    ax.axis("off")
    y = 0.95
    for x, label in ((0.15, "Qty"), (0.5, "Price"), (0.85, "Qty")):
        ax.text(x, y, label, transform=ax.transAxes, color=C["muted"], fontsize=8, ha="center")
    y -= 0.07
    for price, qty in reversed(book.asks):
        ax.text(
            0.5,
            y,
            f"{price:.3f}",
            transform=ax.transAxes,
            color=C["ask"],
            fontsize=8.5,
            ha="center",
        )
        ax.text(0.85, y, str(qty), transform=ax.transAxes, color=C["ask"], fontsize=8, ha="center")
        y -= 0.065
    ax.text(
        0.5,
        y,
        f"── {book.spread:.4f} ──",
        transform=ax.transAxes,
        color=C["muted"],
        fontsize=7,
        ha="center",
    )
    y -= 0.065
    for price, qty in book.bids:
        ax.text(
            0.5,
            y,
            f"{price:.3f}",
            transform=ax.transAxes,
            color=C["bid"],
            fontsize=8.5,
            ha="center",
        )
        ax.text(0.15, y, str(qty), transform=ax.transAxes, color=C["bid"], fontsize=8, ha="center")
        y -= 0.065


def run_live(book: LimitOrderBook, T: float = 120.0, window: float = 60.0) -> FuncAnimation:
    """Animate price, intensities and book until ``T`` simulated seconds."""
    C = THEME
    sim = LiveSimulation(book)

    fig = plt.figure(figsize=(14, 8), facecolor=C["bg"])
    fig.suptitle(
        "Hawkes LOB — live simulation (close the window to stop)", color=C["text"], fontsize=11
    )
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.4, wspace=0.3, left=0.08, right=0.97, top=0.92, bottom=0.08
    )
    ax_price = fig.add_subplot(gs[0, :])
    ax_int = fig.add_subplot(gs[1, 0])
    ax_lob = fig.add_subplot(gs[1, 1])
    style_axis(ax_price, "Mid-price")
    style_axis(ax_int, "Intensities λ(t)")

    (line_mid,) = ax_price.plot([], [], color=C["mid"], lw=1, label="Mid-price")
    (line_lb,) = ax_int.plot([], [], color=C["bid"], lw=0.9, label="λ_bid")
    (line_la,) = ax_int.plot([], [], color=C["ask"], lw=0.9, label="λ_ask")
    for a in (ax_price, ax_int):
        a.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])

    def frames():
        while sim.t < T:
            yield sim.t

    def update(_):
        sim.step(3)
        t, p = list(sim.times), list(sim.prices)
        lb, la = list(sim.lambda_bid), list(sim.lambda_ask)
        line_mid.set_data(t, p)
        line_lb.set_data(t, lb)
        line_la.set_data(t, la)
        for a, arr in ((ax_price, p), (ax_int, lb + la)):
            a.set_xlim(max(0.0, t[-1] - window), t[-1] + 1)
            lo, hi = min(arr), max(arr)
            pad = (hi - lo) * 0.1 + 1e-6
            a.set_ylim(lo - pad, hi + pad)
        _draw_book(ax_lob, sim.book)
        return line_mid, line_lb, line_la

    ani = FuncAnimation(
        fig, update, frames=frames, interval=80, blit=False, cache_frame_data=False, repeat=False
    )
    plt.show()
    return ani
