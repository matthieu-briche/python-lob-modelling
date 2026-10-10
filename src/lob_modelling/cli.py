"""Command-line entry point: ``lob-modelling`` or ``python -m lob_modelling``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .book import LimitOrderBook
from .calibration import mle_bivariate, mle_hawkes
from .hawkes import BID, HawkesParams, HawkesProcess


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lob-modelling", description="Hawkes LOB simulator")
    p.add_argument("--mode", choices=["static", "live", "calibrate"], default="static")
    p.add_argument("--mu", type=float, default=2.0, help="baseline intensity")
    p.add_argument("--alpha", type=float, default=0.6, help="self-excitation")
    p.add_argument("--beta", type=float, default=1.5, help="decay rate")
    p.add_argument("--alpha_x", type=float, default=0.2, help="cross-excitation")
    p.add_argument("--T", type=float, default=120.0, help="simulated horizon (s)")
    p.add_argument("--tick", type=float, default=0.01, help="tick size")
    p.add_argument("--depth", type=int, default=5, help="displayed book depth")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument(
        "--save",
        default="hawkes_lob_results.png",
        help="static mode: output figure path ('' to skip)",
    )
    p.add_argument("--no-show", action="store_true", help="static mode: do not open a window")
    return p


def _banner(args: argparse.Namespace, params: HawkesParams) -> str:
    warn = "" if params.is_stationary else "   (!) non-stationary: intensity will explode"
    return (
        f"Hawkes LOB simulator — mode: {args.mode}\n"
        f"  μ={params.mu}  α={params.alpha}  β={params.beta}  α_x={params.alpha_x}  "
        f"T={args.T}s  branching={params.branching_ratio:.3f}{warn}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        params = HawkesParams(mu=args.mu, alpha=args.alpha, beta=args.beta, alpha_x=args.alpha_x)
    except ValueError as exc:
        build_parser().error(str(exc))
    print(_banner(args, params))

    hawkes_kw = dict(mu=params.mu, alpha=params.alpha, beta=params.beta, alpha_x=params.alpha_x)

    if args.mode == "static":
        if args.no_show:
            import matplotlib

            matplotlib.use("Agg")
        from .analytics import summary_stats
        from .plotting import plot_dashboard

        book = LimitOrderBook(tick_size=args.tick, depth=args.depth, rng=args.seed, **hawkes_kw)
        res = book.run(T=args.T)
        s = summary_stats(res, params)
        print(
            f"  {s['n_events']} events ({s['rate']:.2f} ev/s, "
            f"stationary rate {s['theoretical_rate']:.2f})"
        )
        print(f"  mean λ_bid {s['lambda_bid_mean']:.3f}   mean λ_ask {s['lambda_ask_mean']:.3f}")
        print(f"  price drift {s['drift']:+.4f}")
        plot_dashboard(res, params, save=args.save or None, show=not args.no_show)
        if args.save:
            print(f"  figure saved to {args.save}")

    elif args.mode == "live":
        from .live import run_live

        book = LimitOrderBook(tick_size=args.tick, depth=args.depth, rng=args.seed, **hawkes_kw)
        run_live(book, T=args.T)

    else:  # calibrate
        path = HawkesProcess(rng=args.seed, **hawkes_kw).simulate(args.T)
        bid_times = path.side_times(BID)
        print(f"  {len(path)} events ({len(bid_times)} buys) used for calibration")
        uni = mle_hawkes(bid_times, args.T, mu0=1.0, alpha0=0.4, beta0=1.0)
        bi = mle_bivariate(path.times, path.sides, args.T)
        rows = [
            ("μ", params.mu, uni.mu, bi.mu),
            ("α", params.alpha, uni.alpha, bi.alpha),
            ("α_x", params.alpha_x, None, bi.alpha_x),
            ("β", params.beta, uni.beta, bi.beta),
        ]
        print(f"\n  {'param':<6}{'true':>8}{'univariate':>13}{'bivariate':>12}")
        for name, true, u, b in rows:
            u_s = f"{u:.4f}" if u is not None else "—"
            print(f"  {name:<6}{true:>8.3f}{u_s:>13}{b:>12.4f}")
        print(
            f"\n  converged: univariate {'yes' if uni.success else 'no'}, "
            f"bivariate {'yes' if bi.success else 'no'}"
        )
        if params.alpha_x > 0:
            print("  note: with α_x > 0 the univariate fit on buys alone is misspecified.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
