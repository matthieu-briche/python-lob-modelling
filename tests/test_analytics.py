import numpy as np
import pytest

from lob_modelling import ASK, BID, HawkesProcess, LimitOrderBook
from lob_modelling import analytics as A


def test_ofi_on_hand_built_example():
    times = np.array([0.5, 1.0, 1.5, 4.5])
    sides = np.array([BID, BID, ASK, ASK])
    grid, ofi = A.order_flow_imbalance(times, sides, T=5.0, window=2.0, step=1.0)
    np.testing.assert_allclose(grid, [2.0, 3.0, 4.0])
    # [0,2): 2 buys 1 sell → 1/3 ; [1,3): 1 buy 1 sell → 0 ; [2,4): nothing → 0
    np.testing.assert_allclose(ofi, [1 / 3, 0.0, 0.0])


def test_ofi_bounds():
    path = HawkesProcess(rng=0).simulate(60)
    _, ofi = A.order_flow_imbalance(path.times, path.sides, 60)
    assert np.all(np.abs(ofi) <= 1)


def test_inter_arrival_times():
    np.testing.assert_allclose(A.inter_arrival_times([1.0, 1.5, 3.0]), [0.5, 1.5])
    assert A.inter_arrival_times([1.0]).size == 0


def test_acf_is_normalised_and_flat_for_poisson():
    path = HawkesProcess(mu=5.0, alpha=0.0, beta=1.0, alpha_x=0.0, rng=1).simulate(2000)
    lags, acf = A.count_autocorrelation(path.side_times(BID), 2000, dt=0.2, max_lag=2.0)
    assert lags[0] == 0 and acf[0] == pytest.approx(1.0)
    assert np.all(np.abs(acf[1:]) < 0.05)


def test_acf_positive_for_self_exciting_flow(params):
    hp = HawkesProcess(params.mu, params.alpha, params.beta, params.alpha_x, rng=2)
    path = hp.simulate(1000)
    _, acf = A.count_autocorrelation(path.side_times(BID), 1000, dt=0.2, max_lag=1.0)
    assert acf[1] > 0.05


def test_realized_volatility():
    assert A.realized_volatility([1.0, 1.3, 0.9]) == pytest.approx(np.sqrt(0.09 + 0.16))


def test_summary_stats(params):
    book = LimitOrderBook(rng=5)
    res = book.run(500)
    s = A.summary_stats(res, params)
    assert s["n_buy"] + s["n_sell"] == s["n_events"]
    assert s["rate"] == pytest.approx(s["theoretical_rate"], rel=0.15)
    # Time-averaged intensity is close to the stationary per-side rate.
    assert s["lambda_bid_mean"] == pytest.approx(params.stationary_rate, rel=0.15)
    assert s["stationary"]
