import matplotlib.pyplot as plt
import pytest

from lob_modelling import LimitOrderBook
from lob_modelling.cli import main
from lob_modelling.live import LiveSimulation
from lob_modelling.plotting import plot_dashboard


def test_dashboard_has_six_panels_and_saves(tmp_path, params):
    res = LimitOrderBook(rng=0).run(30)
    out = tmp_path / "dash.png"
    fig = plot_dashboard(res, params, save=out)
    assert len(fig.axes) == 6
    assert out.stat().st_size > 10_000
    plt.close(fig)


def test_dashboard_handles_a_quiet_run(params):
    # Tiny horizon: possibly zero or one event; must not crash.
    res = LimitOrderBook(mu=0.01, alpha=0.0, alpha_x=0.0, rng=0).run(0.5)
    plt.close(plot_dashboard(res, params))


def test_live_simulation_step_records_state():
    sim = LiveSimulation(LimitOrderBook(rng=1), history=10)
    for _ in range(5):
        sim.step(3)
    assert len(sim.times) == 10  # bounded history
    assert list(sim.times) == sorted(sim.times)
    assert sim.t == sim.times[-1]
    assert len(sim.book.bids) == sim.book.depth


def test_cli_static(tmp_path, capsys):
    out = tmp_path / "fig.png"
    assert (
        main(["--mode", "static", "--T", "20", "--seed", "1", "--no-show", "--save", str(out)]) == 0
    )
    assert out.exists()
    assert "events" in capsys.readouterr().out
    plt.close("all")


def test_cli_calibrate(capsys):
    assert main(["--mode", "calibrate", "--T", "200", "--seed", "2"]) == 0
    text = capsys.readouterr().out
    assert "univariate" in text and "bivariate" in text


def test_cli_rejects_invalid_parameters():
    with pytest.raises(SystemExit):
        main(["--beta", "0"])


def test_cli_warns_when_explosive(capsys, tmp_path):
    args = "--mode static --alpha 0.5 --alpha_x 0.1 --beta 0.5 --T 3 --seed 0 --no-show".split()
    main([*args, "--save", ""])
    assert "non-stationary" in capsys.readouterr().out
    plt.close("all")
