import numpy as np
import pytest
from conftest import compensator_increments
from scipy import stats

from lob_modelling import ASK, BID, HawkesParams, HawkesProcess, intensity_on_grid


class TestParams:
    def test_branching_ratio_includes_cross_excitation(self, params):
        assert params.branching_ratio == pytest.approx((0.6 + 0.2) / 1.5)

    def test_stationary_rate(self, params):
        assert params.stationary_rate == pytest.approx(2.0 / (1 - 0.8 / 1.5))

    def test_explosive_regime_is_flagged(self):
        p = HawkesParams(mu=1.0, alpha=1.2, beta=1.5, alpha_x=0.5)
        assert not p.is_stationary
        assert p.stationary_rate == float("inf")

    @pytest.mark.parametrize(
        "kw",
        [dict(mu=0), dict(mu=-1), dict(beta=0), dict(alpha=-0.1), dict(alpha_x=-0.1)],
    )
    def test_invalid_parameters_raise(self, kw):
        with pytest.raises(ValueError):
            HawkesParams(**kw)


class TestSimulation:
    def test_same_seed_same_path(self):
        a = HawkesProcess(rng=7).simulate(50)
        b = HawkesProcess(rng=7).simulate(50)
        np.testing.assert_array_equal(a.times, b.times)
        np.testing.assert_array_equal(a.sides, b.sides)

    def test_different_seeds_differ(self):
        a = HawkesProcess(rng=1).simulate(50)
        b = HawkesProcess(rng=2).simulate(50)
        assert len(a) != len(b) or not np.array_equal(a.times, b.times)

    def test_events_sorted_and_inside_window(self):
        path = HawkesProcess(rng=0).simulate(30)
        assert len(path) > 0
        assert np.all(np.diff(path.times) > 0)
        assert path.times[0] > 0
        assert path.times[-1] <= 30  # regression: an event past T used to be recorded
        assert set(np.unique(path.sides)) <= {BID, ASK}

    def test_simulate_resets_state(self):
        hp = HawkesProcess(rng=3)
        first = hp.simulate(20)
        hp2 = HawkesProcess(rng=3)
        hp2.simulate(5)  # dirty the state
        hp2.rng = np.random.default_rng(3)
        again = hp2.simulate(20)
        np.testing.assert_array_equal(first.times, again.times)

    def test_rejects_non_positive_horizon(self):
        with pytest.raises(ValueError):
            HawkesProcess().simulate(0)

    def test_intensity_before_matches_exact_intensity(self, params):
        hp = HawkesProcess(params.mu, params.alpha, params.beta, params.alpha_x, rng=5)
        path = hp.simulate(40)
        # λ evaluated an instant before each event equals the stored left limit.
        exact = intensity_on_grid(path, params, path.times - 1e-12)
        np.testing.assert_allclose(path.intensity_before, exact, rtol=1e-6)
        assert np.all(path.intensity_before >= params.mu - 1e-12)

    def test_poisson_limit(self):
        hp = HawkesProcess(mu=3.0, alpha=0.0, beta=1.0, alpha_x=0.0, rng=11)
        path = hp.simulate(2000)
        assert len(path.side_times(BID)) / 2000 == pytest.approx(3.0, rel=0.05)
        assert np.allclose(path.intensity_before, 3.0)

    def test_mean_rate_matches_stationary_rate(self, params):
        hp = HawkesProcess(params.mu, params.alpha, params.beta, params.alpha_x, rng=21)
        T = 3000
        path = hp.simulate(T)
        for side in (BID, ASK):
            rate = len(path.side_times(side)) / T
            assert rate == pytest.approx(params.stationary_rate, rel=0.08)

    def test_time_rescaling_gives_unit_exponentials(self, params):
        """Goodness of fit of the thinning sampler (Brown et al., 2002)."""
        hp = HawkesProcess(params.mu, params.alpha, params.beta, params.alpha_x, rng=4)
        path = hp.simulate(1000)
        for inc in compensator_increments(path.times, path.sides, params):
            assert stats.kstest(inc, "expon").pvalue > 0.01


class TestIntensityGrid:
    def test_no_events_is_baseline(self, params):
        path = HawkesProcess(rng=0).simulate(1e-9)
        lam = intensity_on_grid(path, params, np.linspace(0, 1, 5))
        np.testing.assert_allclose(lam, params.mu)

    def test_single_event_decays_exponentially(self, params):
        from lob_modelling.hawkes import HawkesPath

        path = HawkesPath(
            times=np.array([1.0]),
            sides=np.array([BID]),
            intensity_before=np.array([[params.mu, params.mu]]),
            T=5.0,
        )
        lam = intensity_on_grid(path, params, np.array([0.5, 1.0, 2.0]))
        d = np.exp(-params.beta * 1.0)
        np.testing.assert_allclose(lam[0], [params.mu, params.mu])
        np.testing.assert_allclose(lam[1], [params.mu + params.alpha, params.mu + params.alpha_x])
        np.testing.assert_allclose(
            lam[2], [params.mu + params.alpha * d, params.mu + params.alpha_x * d]
        )
