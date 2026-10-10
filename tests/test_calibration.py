import numpy as np
import pytest
from scipy.integrate import quad

from lob_modelling import (
    BID,
    HawkesProcess,
    bivariate_loglik,
    hawkes_loglik,
    mle_bivariate,
    mle_hawkes,
)


def brute_force_loglik(t, T, mu, alpha, beta):
    """O(n²) log-likelihood with a numerically integrated compensator."""

    def lam(s):
        past = t[t < s]
        return mu + alpha * np.exp(-beta * (s - past)).sum()

    log_term = sum(np.log(lam(ti)) for ti in t)
    breakpoints = np.concatenate([[0.0], t, [T]])
    integral = sum(
        quad(lam, a, b)[0] for a, b in zip(breakpoints[:-1], breakpoints[1:], strict=True)
    )
    return log_term - integral


def test_univariate_loglik_matches_brute_force():
    t = np.array([0.3, 0.5, 1.4, 1.45, 3.0])
    got = hawkes_loglik(t, 4.0, 1.2, 0.7, 2.0)
    assert got == pytest.approx(brute_force_loglik(t, 4.0, 1.2, 0.7, 2.0), rel=1e-6)


def test_bivariate_without_cross_term_splits_into_two_univariate():
    path = HawkesProcess(mu=1.5, alpha=0.5, beta=2.0, alpha_x=0.0, rng=0).simulate(50)
    bi = bivariate_loglik(path.times, path.sides, 50, 1.5, 0.5, 0.0, 2.0)
    uni = sum(hawkes_loglik(path.side_times(s), 50, 1.5, 0.5, 2.0) for s in (0, 1))
    assert bi == pytest.approx(uni, rel=1e-10)


def test_true_parameters_beat_wrong_ones(params):
    path = HawkesProcess(params.mu, params.alpha, params.beta, params.alpha_x, rng=3).simulate(300)
    args = (path.times, path.sides, 300)
    truth = bivariate_loglik(*args, params.mu, params.alpha, params.alpha_x, params.beta)
    assert truth > bivariate_loglik(*args, params.mu, params.alpha, 0.0, params.beta)
    assert truth > bivariate_loglik(*args, 2 * params.mu, params.alpha, params.alpha_x, params.beta)


@pytest.mark.slow
def test_univariate_mle_recovers_parameters():
    path = HawkesProcess(mu=2.0, alpha=0.6, beta=1.5, alpha_x=0.0, rng=12).simulate(2500)
    fit = mle_hawkes(path.side_times(BID), 2500, mu0=1.0, alpha0=0.4, beta0=1.0)
    assert fit.success
    assert fit.mu == pytest.approx(2.0, rel=0.15)
    assert fit.alpha == pytest.approx(0.6, rel=0.15)
    assert fit.beta == pytest.approx(1.5, rel=0.2)


@pytest.mark.slow
def test_bivariate_mle_recovers_cross_excitation(params):
    T = 2000
    path = HawkesProcess(params.mu, params.alpha, params.beta, params.alpha_x, rng=8).simulate(T)
    fit = mle_bivariate(path.times, path.sides, T)
    assert fit.success
    assert fit.mu == pytest.approx(params.mu, rel=0.15)
    assert fit.alpha == pytest.approx(params.alpha, rel=0.15)
    assert fit.alpha_x == pytest.approx(params.alpha_x, abs=0.08)
    assert fit.beta == pytest.approx(params.beta, rel=0.2)
    assert fit.branching_ratio == pytest.approx(params.branching_ratio, rel=0.1)


@pytest.mark.parametrize(
    "times, T",
    [([0.5, 0.2], 1.0), ([0.2, 1.5], 1.0), ([-0.1, 0.5], 1.0), ([0.1, 0.2], 0.0)],
)
def test_invalid_event_times_raise(times, T):
    with pytest.raises(ValueError):
        hawkes_loglik(times, T, 1.0, 0.5, 1.0)


def test_too_few_events_raise():
    with pytest.raises(ValueError):
        mle_hawkes([0.5], 1.0)
