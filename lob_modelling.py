"""
Simulateur de Carnet d'Ordres par Processus de Hawkes Bivarié
=============================================================
Modèle : Hawkes mutuellement excitant avec noyau exponentiel
    λ_i(t) = μ + Σ_j Σ_{t_k^j < t} α_ij · exp(-β(t - t_k^j))

Simulation par méthode de thinning (Ogata 1981)

Dépendances :
    pip install numpy matplotlib scipy

Usage :
    python hawkes_lob_simulator.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
from collections import deque
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────
# 1.  PROCESSUS DE HAWKES BIVARIÉ
# ─────────────────────────────────────────────

class HawkesProcess:
    """
    Hawkes bivarié (bid=0, ask=1) à noyau exponentiel.

    Paramètres
    ----------
    mu       : float  – intensité de base (events/s)
    alpha    : float  – auto-excitabilité  (< beta pour stationnarité)
    beta     : float  – taux de décroissance du noyau
    alpha_x  : float  – excitation croisée bid→ask et ask→bid
    """

    def __init__(self, mu=2.0, alpha=0.6, beta=1.5, alpha_x=0.2):
        self.mu      = mu
        self.alpha   = alpha
        self.beta    = beta
        self.alpha_x = alpha_x

        # État courant : R_ij = Σ_{t_k^j} α_ij exp(-β(t - t_k^j))
        # i = processus affecté, j = processus source
        self.R = np.zeros((2, 2))   # R[i,j]
        self.t = 0.0
        self.event_times = [[], []]  # historique

    def intensities(self):
        """Retourne (λ_bid, λ_ask) au temps courant."""
        lam = self.mu + self.R[0, 0] + self.R[0, 1], \
              self.mu + self.R[1, 1] + self.R[1, 0]
        return max(lam[0], 1e-9), max(lam[1], 1e-9)

    def _decay(self, dt):
        """Décroît les excitations de dt secondes."""
        self.R *= np.exp(-self.beta * dt)

    def _trigger(self, process: int):
        """Enregistre un événement sur le processus `process` (0=bid, 1=ask)."""
        j = process
        self.R[j, j]     += self.alpha        # auto-excitation
        self.R[1-j, j]   += self.alpha_x      # excitation croisée
        self.event_times[j].append(self.t)

    def next_event(self):
        """
        Génère le prochain événement (méthode de thinning d'Ogata).
        Retourne (t_event, process) où process ∈ {0,1}.
        """
        while True:
            lb, la = self.intensities()
            lam_bar = lb + la + 1e-6   # borne supérieure homogène

            # Temps candidate ~ Exp(lam_bar)
            dt = np.random.exponential(1.0 / lam_bar)
            self._decay(dt)
            self.t += dt

            # Acceptation / rejet
            lb2, la2 = self.intensities()
            u = np.random.uniform(0, lam_bar)
            if u < lb2:
                self._trigger(0)
                return self.t, 0
            elif u < lb2 + la2:
                self._trigger(1)
                return self.t, 1
            # sinon : fantôme, on recommence

    def simulate(self, T: float):
        """
        Simule jusqu'au temps T.
        Retourne (times, processes) arrays.
        """
        self.t = 0.0
        self.R[:] = 0.0
        self.event_times = [[], []]
        times, procs = [], []

        while self.t < T:
            t_ev, proc = self.next_event()
            if t_ev > T:
                break
            times.append(t_ev)
            procs.append(proc)

        return np.array(times), np.array(procs)


# ─────────────────────────────────────────────
# 2.  CARNET D'ORDRES SIMULÉ
# ─────────────────────────────────────────────

class LimitOrderBook:
    """
    LOB simplifié dont le mid-price évolue selon les market orders
    générés par le processus de Hawkes.
    """

    def __init__(self, mid_price=100.0, tick_size=0.01, depth=5,
                 mu=2.0, alpha=0.6, beta=1.5, alpha_x=0.2):
        self.mid0       = mid_price
        self.mid_price  = mid_price
        self.tick       = tick_size
        self.depth      = depth
        self.hawkes     = HawkesProcess(mu, alpha, beta, alpha_x)

        # Historiques
        self.price_hist   = deque(maxlen=500)
        self.lambda_b_hist = deque(maxlen=500)
        self.lambda_a_hist = deque(maxlen=500)
        self.time_hist    = deque(maxlen=500)
        self.events       = deque(maxlen=200)  # (t, side, price)

        self._build_lob()

    # ── LOB snapshot ──────────────────────────

    def _build_lob(self):
        spread = self.tick * 2
        best_ask = self.mid_price + spread / 2
        best_bid = self.mid_price - spread / 2

        self.bids = [
            (round(best_bid - i * self.tick, 6),
             int(np.random.randint(10, 100) + (self.depth - i) * 8))
            for i in range(self.depth)
        ]
        self.asks = [
            (round(best_ask + i * self.tick, 6),
             int(np.random.randint(10, 100) + (self.depth - i) * 8))
            for i in range(self.depth)
        ]

    # ── Simulation en batch ───────────────────

    def run(self, T=60.0):
        """
        Simule T secondes et stocke tous les états.
        Retourne un dict de résultats pour analyse.
        """
        times, procs = self.hawkes.simulate(T)
        self.mid_price = self.mid0
        self.hawkes.t  = 0.0
        self.hawkes.R[:] = 0.0
        self.hawkes.event_times = [[], []]   # reconstruit par _trigger ci-dessous

        mid_prices = [self.mid0]
        t_axis     = [0.0]
        lambda_b   = [self.hawkes.mu]
        lambda_a   = [self.hawkes.mu]

        # Rejeu des événements : on fait décroître puis exciter l'état R
        # pour reconstruire λ(t-) juste avant chaque événement.
        for t, p in zip(times, procs):
            self.hawkes._decay(t - self.hawkes.t)
            self.hawkes.t = t
            lb, la = self.hawkes.intensities()
            self.hawkes._trigger(int(p))

            if p == 0:   # bid market order → prix monte
                self.mid_price += self.tick * np.random.uniform(0.2, 0.8)
            else:        # ask market order → prix baisse
                self.mid_price -= self.tick * np.random.uniform(0.2, 0.8)

            mid_prices.append(self.mid_price)
            t_axis.append(t)
            lambda_b.append(lb)
            lambda_a.append(la)

        return {
            "times"       : np.array(t_axis),
            "mid_prices"  : np.array(mid_prices),
            "event_times" : times,
            "event_sides" : procs,
            "lambda_bid"  : np.array(lambda_b),
            "lambda_ask"  : np.array(lambda_a),
        }


# ─────────────────────────────────────────────
# 3.  VISUALISATION STATIQUE (analyse batch)
# ─────────────────────────────────────────────

def plot_simulation(res, params: dict, T=60.0):
    """
    Dashboard complet : mid-price, intensités, distribution inter-arrivées,
    clustering des événements.
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle(
        f"Hawkes LOB Simulator  |  μ={params['mu']}  α={params['alpha']}  "
        f"β={params['beta']}  α_x={params['alpha_x']}",
        color="#e6edf3", fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    C = {"bg": "#0d1117", "panel": "#161b22", "border": "#30363d",
         "bid": "#3fb950", "ask": "#f85149", "mid": "#58a6ff",
         "text": "#e6edf3", "muted": "#8b949e", "accent": "#d2a8ff"}

    def ax(pos, colspan=1, rowspan=1):
        a = fig.add_subplot(gs[pos[0]:pos[0]+rowspan, pos[1]:pos[1]+colspan])
        a.set_facecolor(C["panel"])
        for sp in a.spines.values():
            sp.set_color(C["border"])
        a.tick_params(colors=C["muted"], labelsize=8)
        a.xaxis.label.set_color(C["muted"])
        a.yaxis.label.set_color(C["muted"])
        return a

    times   = res["times"]
    mids    = res["mid_prices"]
    ev_t    = res["event_times"]
    ev_s    = res["event_sides"]
    lb      = res["lambda_bid"]
    la      = res["lambda_ask"]

    bid_mask = ev_s == 0
    ask_mask = ev_s == 1

    # ── Panel 1 : Mid-price ──────────────────
    a1 = ax((0, 0), colspan=3)
    a1.plot(times, mids, color=C["mid"], lw=0.9, label="Mid-price")
    a1.scatter(ev_t[bid_mask], mids[np.searchsorted(times, ev_t[bid_mask])],
               color=C["bid"], s=12, alpha=0.5, label="Buy MO", zorder=3)
    a1.scatter(ev_t[ask_mask], mids[np.searchsorted(times, ev_t[ask_mask])],
               color=C["ask"], s=12, alpha=0.5, label="Sell MO", zorder=3)
    a1.fill_between(times, mids, mids.min(), alpha=0.08, color=C["mid"])
    a1.set_ylabel("Prix", fontsize=8)
    a1.set_xlabel("Temps (s)", fontsize=8)
    a1.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"], framealpha=0.6)
    a1.set_title("Mid-price & Market Orders", color=C["text"], fontsize=9, pad=6)

    # ── Panel 2 : Intensités ─────────────────
    a2 = ax((1, 0), colspan=2)
    a2.plot(times, lb, color=C["bid"], lw=0.8, label="λ_bid(t)")
    a2.plot(times, la, color=C["ask"], lw=0.8, label="λ_ask(t)", alpha=0.8)
    a2.axhline(params["mu"], color=C["muted"], lw=0.6, ls="--", label="μ baseline")
    a2.fill_between(times, lb, params["mu"], alpha=0.1, color=C["bid"])
    a2.fill_between(times, la, params["mu"], alpha=0.1, color=C["ask"])
    a2.set_ylabel("Intensité (ev/s)", fontsize=8)
    a2.set_xlabel("Temps (s)", fontsize=8)
    a2.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"], framealpha=0.6)
    a2.set_title("Intensités instantanées", color=C["text"], fontsize=9, pad=6)

    # ── Panel 3 : Order flow imbalance ───────
    a3 = ax((1, 2))
    # OFI glissant sur fenêtre de 5s
    window = 5.0
    ofi_vals, ofi_t = [], []
    for t in np.arange(window, T, 0.5):
        mask = (ev_t >= t - window) & (ev_t < t)
        n_bid = bid_mask[mask].sum()
        n_ask = ask_mask[mask].sum()
        ofi = (n_bid - n_ask) / max(n_bid + n_ask, 1)
        ofi_vals.append(ofi)
        ofi_t.append(t)
    ofi_vals = np.array(ofi_vals)
    colors_ofi = [C["bid"] if v >= 0 else C["ask"] for v in ofi_vals]
    a3.bar(ofi_t, ofi_vals, width=0.45, color=colors_ofi, alpha=0.8)
    a3.axhline(0, color=C["border"], lw=0.5)
    a3.set_ylim(-1, 1)
    a3.set_ylabel("OFI", fontsize=8)
    a3.set_xlabel("Temps (s)", fontsize=8)
    a3.set_title("Order Flow Imbalance", color=C["text"], fontsize=9, pad=6)

    # ── Panel 4 : Distribution inter-arrivées ─
    a4 = ax((2, 0))
    iat_bid = np.diff(ev_t[bid_mask]) if bid_mask.sum() > 1 else np.array([1.0])
    iat_ask = np.diff(ev_t[ask_mask]) if ask_mask.sum() > 1 else np.array([1.0])
    bins = np.linspace(0, np.percentile(np.concatenate([iat_bid, iat_ask]), 95), 30)
    a4.hist(iat_bid, bins=bins, color=C["bid"], alpha=0.6, label="bid", density=True)
    a4.hist(iat_ask, bins=bins, color=C["ask"], alpha=0.4, label="ask", density=True)
    # Théorique Exp(mu) pour comparaison
    x = np.linspace(0, bins[-1], 100)
    a4.plot(x, params["mu"] * np.exp(-params["mu"] * x),
            color=C["muted"], lw=1, ls="--", label="Exp(μ)")
    a4.set_xlabel("Inter-arrivée (s)", fontsize=8)
    a4.set_ylabel("Densité", fontsize=8)
    a4.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"], framealpha=0.6)
    a4.set_title("Distribution inter-arrivées", color=C["text"], fontsize=9, pad=6)

    # ── Panel 5 : Fonction d'autocorrélation ─
    a5 = ax((2, 1))
    # Processus ponctuel → proxy binaire sur grille
    dt_grid = 0.2
    t_grid = np.arange(0, T, dt_grid)
    n_bid_grid = np.array([((ev_t[bid_mask] >= t) & (ev_t[bid_mask] < t+dt_grid)).sum()
                           for t in t_grid], dtype=float)
    from numpy.fft import fft, ifft
    n_ = n_bid_grid - n_bid_grid.mean()
    acf = np.real(ifft(fft(n_) * np.conj(fft(n_))))
    acf /= max(acf[0], 1e-9)
    lags = np.arange(len(acf)) * dt_grid
    max_lag = min(20, len(acf) // 2)
    a5.plot(lags[:max_lag], acf[:max_lag], color=C["bid"], lw=1)
    a5.axhline(0, color=C["border"], lw=0.5)
    a5.fill_between(lags[:max_lag], acf[:max_lag], 0, alpha=0.12, color=C["bid"])
    a5.set_xlabel("Lag (s)", fontsize=8)
    a5.set_ylabel("ACF", fontsize=8)
    a5.set_title("Autocorrélation λ_bid", color=C["text"], fontsize=9, pad=6)

    # ── Panel 6 : Stats summary ───────────────
    a6 = ax((2, 2))
    a6.axis("off")
    n_ev = len(ev_t)
    n_b  = bid_mask.sum()
    n_a  = ask_mask.sum()
    drift = mids[-1] - mids[0]
    vol   = np.std(np.diff(mids)) * np.sqrt(len(mids))

    stats = [
        ("Événements totaux",    f"{n_ev}"),
        ("Buy MO",               f"{n_b}  ({100*n_b/max(n_ev,1):.0f}%)"),
        ("Sell MO",              f"{n_a}  ({100*n_a/max(n_ev,1):.0f}%)"),
        ("Taux moyen (ev/s)",    f"{n_ev/T:.2f}"),
        ("λ_bid moyen",          f"{lb.mean():.3f}"),
        ("λ_ask moyen",          f"{la.mean():.3f}"),
        ("Drift prix",           f"{drift:+.4f}"),
        ("Volatilité réalisée",  f"{vol:.4f}"),
        ("Branch. ratio α/β",    f"{params['alpha']/params['beta']:.3f}"),
        ("Stationnarité",        "✓" if params['alpha'] < params['beta'] else "✗ instable"),
    ]
    y0 = 0.97
    for label, val in stats:
        a6.text(0.02, y0, label, transform=a6.transAxes, fontsize=7.5,
                color=C["muted"])
        a6.text(0.98, y0, val,   transform=a6.transAxes, fontsize=7.5,
                color=C["text"], ha="right", fontweight="bold")
        y0 -= 0.095
    a6.set_title("Statistiques", color=C["text"], fontsize=9, pad=6)

    plt.savefig("hawkes_lob_results.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    print("✓ Figure sauvegardée : hawkes_lob_results.png")
    plt.show()


# ─────────────────────────────────────────────
# 4.  ANIMATION EN TEMPS RÉEL (optionnelle)
# ─────────────────────────────────────────────

def run_live(params: dict, T_live=120.0):
    """
    Simulation en temps réel avec animation matplotlib.
    Le LOB, le mid-price et les intensités se mettent à jour en direct.
    """
    lob = LimitOrderBook(**params)

    fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")
    fig.suptitle("Hawkes LOB — Simulation en temps réel  (ferme la fenêtre pour arrêter)",
                 color="#e6edf3", fontsize=11)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                           left=0.08, right=0.97, top=0.92, bottom=0.08)
    C = {"bg":"#0d1117","panel":"#161b22","border":"#30363d",
         "bid":"#3fb950","ask":"#f85149","mid":"#58a6ff","text":"#e6edf3","muted":"#8b949e"}

    def styled_ax(pos):
        a = fig.add_subplot(gs[pos])
        a.set_facecolor(C["panel"])
        for sp in a.spines.values(): sp.set_color(C["border"])
        a.tick_params(colors=C["muted"], labelsize=8)
        return a

    ax_price = styled_ax((0, slice(0, 2)))
    ax_int   = styled_ax((1, 0))
    ax_lob   = styled_ax((1, 1))

    # Lignes vides
    line_mid,  = ax_price.plot([], [], color=C["mid"],  lw=1,   label="Mid-price")
    line_lb,   = ax_int.plot  ([], [], color=C["bid"],  lw=0.9, label="λ_bid")
    line_la,   = ax_int.plot  ([], [], color=C["ask"],  lw=0.9, label="λ_ask")

    for a in [ax_price, ax_int, ax_lob]:
        a.tick_params(colors=C["muted"], labelsize=8)
    ax_price.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])
    ax_int.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["text"])
    ax_price.set_title("Mid-price", color=C["text"], fontsize=9)
    ax_int.set_title("Intensités λ(t)", color=C["text"], fontsize=9)
    ax_lob.set_title("LOB snapshot", color=C["text"], fontsize=9)
    ax_lob.axis("off")

    SIM_DT = 0.15   # secondes simulées par frame

    def update(frame):
        # Avancer la simulation
        for _ in range(3):
            t_ev, proc = lob.hawkes.next_event()
            if proc == 0:
                lob.mid_price += lob.tick * np.random.uniform(0.2, 0.7)
            else:
                lob.mid_price -= lob.tick * np.random.uniform(0.2, 0.7)
            lb_, la_ = lob.hawkes.intensities()
            lob.price_hist.append(lob.mid_price)
            lob.lambda_b_hist.append(lb_)
            lob.lambda_a_hist.append(la_)
            lob.time_hist.append(lob.hawkes.t)
            lob.events.append((lob.hawkes.t, proc, lob.mid_price))

        t_arr = list(lob.time_hist)
        p_arr = list(lob.price_hist)
        lb_arr = list(lob.lambda_b_hist)
        la_arr = list(lob.lambda_a_hist)

        line_mid.set_data(t_arr, p_arr)
        line_lb.set_data(t_arr, lb_arr)
        line_la.set_data(t_arr, la_arr)

        for a, arr in [(ax_price, p_arr), (ax_int, lb_arr + la_arr)]:
            a.set_xlim(max(0, t_arr[-1]-60), t_arr[-1]+1)
            mn, mx = min(arr), max(arr)
            pad = (mx - mn) * 0.1 + 1e-6
            a.set_ylim(mn - pad, mx + pad)

        # LOB snapshot
        ax_lob.cla()
        ax_lob.set_facecolor(C["panel"])
        for sp in ax_lob.spines.values(): sp.set_color(C["border"])
        ax_lob.axis("off")
        lob._build_lob()
        spread = lob.tick * 2
        best_ask = lob.mid_price + spread / 2
        best_bid = lob.mid_price - spread / 2
        y = 0.95
        ax_lob.text(0.15, y, "Qty", transform=ax_lob.transAxes,
                    color=C["muted"], fontsize=8, ha="center")
        ax_lob.text(0.5,  y, "Prix", transform=ax_lob.transAxes,
                    color=C["muted"], fontsize=8, ha="center")
        ax_lob.text(0.85, y, "Qty", transform=ax_lob.transAxes,
                    color=C["muted"], fontsize=8, ha="center")
        y -= 0.07
        for price, qty in reversed(lob.asks):
            ax_lob.text(0.5,  y, f"{price:.3f}", transform=ax_lob.transAxes,
                        color=C["ask"], fontsize=8.5, ha="center")
            ax_lob.text(0.85, y, str(qty), transform=ax_lob.transAxes,
                        color=C["ask"], fontsize=8, ha="center")
            y -= 0.065
        ax_lob.text(0.5, y, f"── {spread:.4f} ──", transform=ax_lob.transAxes,
                    color=C["muted"], fontsize=7, ha="center")
        y -= 0.065
        for price, qty in lob.bids:
            ax_lob.text(0.5,  y, f"{price:.3f}", transform=ax_lob.transAxes,
                        color=C["bid"], fontsize=8.5, ha="center")
            ax_lob.text(0.15, y, str(qty), transform=ax_lob.transAxes,
                        color=C["bid"], fontsize=8, ha="center")
            y -= 0.065
        ax_lob.set_title("LOB snapshot", color=C["text"], fontsize=9)

        return line_mid, line_lb, line_la

    ani = FuncAnimation(fig, update, interval=80, blit=False, cache_frame_data=False)
    plt.show()


# ─────────────────────────────────────────────
# 5.  CALIBRATION MLE (bonus)
# ─────────────────────────────────────────────

def mle_hawkes(event_times, T, mu0=1.0, alpha0=0.5, beta0=2.0):
    """
    Calibration MLE d'un Hawkes univarié par L-BFGS-B.
    Log-vraisemblance de Ozaki (1979).
    """
    from scipy.optimize import minimize

    def log_likelihood(params):
        mu_, alpha_, beta_ = params
        if mu_ <= 0 or alpha_ <= 0 or beta_ <= 0 or alpha_ >= beta_:
            return 1e10
        t = event_times
        n = len(t)

        # Termes d'intensité aux événements
        R = np.zeros(n)
        for i in range(1, n):
            R[i] = np.exp(-beta_ * (t[i] - t[i-1])) * (1 + R[i-1])

        lam_at_events = mu_ + alpha_ * R
        ll = np.sum(np.log(lam_at_events + 1e-300))

        # Terme intégral ∫ λ(t) dt
        A = np.sum(1 - np.exp(-beta_ * (T - t)))
        integral = mu_ * T + alpha_ / beta_ * A

        return -(ll - integral)

    res = minimize(log_likelihood, [mu0, alpha0, beta0],
                   method="L-BFGS-B",
                   bounds=[(1e-4, None), (1e-4, None), (1e-4, None)],
                   options={"maxiter": 500, "ftol": 1e-10})
    return {"mu": res.x[0], "alpha": res.x[1], "beta": res.x[2],
            "success": res.success, "ll": -res.fun}


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hawkes LOB Simulator")
    parser.add_argument("--mode",    choices=["static", "live", "calibrate"],
                        default="static", help="Mode de lancement")
    parser.add_argument("--mu",      type=float, default=2.0,  help="Baseline intensity")
    parser.add_argument("--alpha",   type=float, default=0.6,  help="Self-excitation")
    parser.add_argument("--beta",    type=float, default=1.5,  help="Decay rate")
    parser.add_argument("--alpha_x", type=float, default=0.2,  help="Cross-excitation")
    parser.add_argument("--T",       type=float, default=120.0, help="Durée (s)")
    parser.add_argument("--tick",    type=float, default=0.01, help="Tick size")
    parser.add_argument("--depth",   type=int,   default=5,    help="Profondeur LOB")
    parser.add_argument("--seed",    type=int,   default=None, help="Graine aléatoire (reproductibilité)")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    params = dict(mu=args.mu, alpha=args.alpha, beta=args.beta,
                  alpha_x=args.alpha_x, mid_price=100.0,
                  tick_size=args.tick, depth=args.depth)

    print(f"""
╔══════════════════════════════════════════════╗
║   Hawkes LOB Simulator  —  mode: {args.mode:<10}║
╠══════════════════════════════════════════════╣
║  μ={args.mu}  α={args.alpha}  β={args.beta}  α_x={args.alpha_x}  T={args.T}s  ║
╚══════════════════════════════════════════════╝
""")

    if args.mode == "static":
        print("Simulation en cours...")
        lob = LimitOrderBook(**params)
        res = lob.run(T=args.T)
        n = len(res["event_times"])
        print(f"✓ {n} événements générés en {args.T}s  (taux moyen: {n/args.T:.2f} ev/s)")
        print(f"  λ_bid moyen : {res['lambda_bid'].mean():.3f}")
        print(f"  λ_ask moyen : {res['lambda_ask'].mean():.3f}")
        print(f"  Drift prix  : {res['mid_prices'][-1]-res['mid_prices'][0]:+.4f}")
        plot_simulation(res, params, T=args.T)

    elif args.mode == "live":
        print("Animation en temps réel — ferme la fenêtre pour arrêter.")
        run_live(params, T_live=args.T)

    elif args.mode == "calibrate":
        print("Génération de données synthétiques puis calibration MLE...")
        hp = HawkesProcess(mu=args.mu, alpha=args.alpha, beta=args.beta,
                           alpha_x=args.alpha_x)
        times, procs = hp.simulate(args.T)
        bid_times = times[procs == 0]
        print(f"  {len(bid_times)} événements bid pour la calibration")

        res_mle = mle_hawkes(bid_times, args.T, mu0=1.0, alpha0=0.4, beta0=1.0)
        print(f"""
  Vrais paramètres  : μ={args.mu}  α={args.alpha}  β={args.beta}
  Estimés MLE       : μ={res_mle['mu']:.4f}  α={res_mle['alpha']:.4f}  β={res_mle['beta']:.4f}
  Log-vraisemblance : {res_mle['ll']:.2f}
  Convergence       : {'✓' if res_mle['success'] else '✗'}
""")