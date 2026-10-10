import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lob_modelling import HawkesParams  # noqa: E402


@pytest.fixture
def params() -> HawkesParams:
    return HawkesParams(mu=2.0, alpha=0.6, beta=1.5, alpha_x=0.2)


def compensator_increments(times, sides, p: HawkesParams):
    """Λ_i between consecutive events of side i (time-rescaling theorem).

    For a correctly simulated point process these are i.i.d. Exp(1).
    """
    R = np.zeros(2)  # current excitation of each side's intensity
    Lam = np.zeros(2)  # running compensator of each side
    last = [0.0, 0.0]
    t_prev = 0.0
    out = [[], []]
    for t, s in zip(times, sides, strict=True):
        dt = t - t_prev
        Lam += p.mu * dt + R * (1 - np.exp(-p.beta * dt)) / p.beta
        R *= np.exp(-p.beta * dt)
        t_prev = t
        out[s].append(Lam[s] - last[s])
        last[s] = Lam[s]
        R[s] += p.alpha
        R[1 - s] += p.alpha_x
    return np.array(out[0]), np.array(out[1])
