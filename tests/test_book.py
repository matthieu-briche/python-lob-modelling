import numpy as np
import pytest

from lob_modelling import ASK, BID, LimitOrderBook


def test_buy_lifts_and_sell_lowers_mid():
    book = LimitOrderBook(mid_price=100, tick_size=0.01, impact=(0.2, 0.8), rng=0)
    up = book.apply_market_order(BID) - 100
    assert 0.002 - 1e-12 <= up <= 0.008 + 1e-12
    before = book.mid_price
    down = before - book.apply_market_order(ASK)
    assert 0.002 - 1e-12 <= down <= 0.008 + 1e-12


def test_run_shapes_and_consistency():
    book = LimitOrderBook(rng=1)
    res = book.run(T=30)
    n = len(res.event_times)
    assert n > 0
    for arr in (res.times, res.mid_prices, res.lambda_bid, res.lambda_ask):
        assert arr.shape == (n + 1,)
    assert res.times[0] == 0 and res.mid_prices[0] == 100
    # Each price step has the sign of its order and a bounded size.
    steps = np.diff(res.mid_prices)
    assert np.all(np.sign(steps) == np.where(res.event_sides == BID, 1, -1))
    assert np.all(np.abs(steps) <= book.tick * book.impact[1] + 1e-12)


def test_run_is_reproducible():
    a = LimitOrderBook(rng=9).run(20)
    b = LimitOrderBook(rng=9).run(20)
    np.testing.assert_array_equal(a.mid_prices, b.mid_prices)


def test_levels_bracket_the_mid():
    book = LimitOrderBook(mid_price=50, tick_size=0.05, depth=4, rng=2)
    assert len(book.bids) == len(book.asks) == 4
    best_bid, best_ask = book.bids[0][0], book.asks[0][0]
    assert best_ask - best_bid == pytest.approx(book.spread)
    assert best_bid < book.mid_price < best_ask
    assert [p for p, _ in book.bids] == sorted((p for p, _ in book.bids), reverse=True)
    assert [p for p, _ in book.asks] == sorted(p for p, _ in book.asks)
    assert all(q > 0 for _, q in book.bids + book.asks)


def test_refresh_without_churn_keeps_queues():
    book = LimitOrderBook(rng=3)
    before = [q for _, q in book.bids + book.asks]
    book.refresh_levels(churn=0.0)
    assert [q for _, q in book.bids + book.asks] == before


@pytest.mark.parametrize(
    "kw", [dict(tick_size=0), dict(depth=0), dict(impact=(0.5, 0.1)), dict(impact=(-0.1, 0.5))]
)
def test_invalid_book_arguments(kw):
    with pytest.raises(ValueError):
        LimitOrderBook(**kw)
