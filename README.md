# LOB Modelling — Hawkes-driven limit order book simulator

Simulation of a limit order book (LOB) whose market order flow follows a **bivariate Hawkes process** (buy / sell) with an exponential kernel, with an analysis dashboard, real-time animation and maximum likelihood calibration.

![Simulation dashboard](docs/hawkes_lob_results.png)

## Model

Intensity of process *i* ∈ {bid, ask}:

$$\lambda_i(t) = \mu + \sum_{j} \sum_{t_k^j < t} \alpha_{ij}\, e^{-\beta (t - t_k^j)}$$

- `μ`: baseline intensity (events/s)
- `α`: self-excitation (a buy order triggers further buy orders)
- `α_x`: cross-excitation (buy → sell and sell → buy)
- `β`: kernel decay rate

The process is stationary when the branching ratio `(α + α_x) / β < 1`.

Simulation uses **Ogata's thinning algorithm (1981)**; calibration relies on **Ozaki's (1979)** recursive log-likelihood, optimised with L-BFGS-B.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Static dashboard (mid-price, intensities, OFI, inter-arrival times, ACF, stats)
python lob_modelling.py --mode static --T 120 --seed 42

# Real-time LOB animation
python lob_modelling.py --mode live

# Synthetic data then MLE parameter estimation
python lob_modelling.py --mode calibrate --T 1500 --alpha_x 0 --seed 1
```

| Option      | Default | Description              |
|-------------|---------|--------------------------|
| `--mu`      | 2.0     | Baseline intensity       |
| `--alpha`   | 0.6     | Self-excitation          |
| `--beta`    | 1.5     | Decay rate               |
| `--alpha_x` | 0.2     | Cross-excitation         |
| `--T`       | 120     | Simulated horizon (s)    |
| `--tick`    | 0.01    | Tick size                |
| `--depth`   | 5       | Displayed book depth     |
| `--seed`    | none    | Random seed (reproducible runs) |

## Code structure

| Component           | Content                                                   |
|---------------------|-----------------------------------------------------------|
| `HawkesProcess`     | Bivariate Hawkes, Ogata thinning, simulation on `[0, T]`  |
| `LimitOrderBook`    | Stylised book; mid-price reacts to market orders          |
| `plot_simulation`   | Six-panel analysis dashboard                              |
| `run_live`          | Matplotlib animation of price, intensities and book       |
| `mle_hawkes`        | MLE calibration of a univariate Hawkes process            |

## Calibration check

With `--alpha_x 0 --T 1500 --seed 1` (about 5,000 buy events), the MLE recovers the true parameters:

| Parameter | True | Estimated |
|-----------|------|-----------|
| μ         | 2.0  | 1.99      |
| α         | 0.6  | 0.62      |
| β         | 1.5  | 1.55      |

## Known limitations

- Calibration is **univariate**: with cross-excitation (`α_x` ≠ 0) the bid stream is no longer a univariate Hawkes process and the estimates of `α` and `β` are biased.
- The book is stylised: fixed 2-tick spread, random queue sizes, uniform price impact between 0.2 and 0.8 tick.

## References

- Ogata, Y. (1981). *On Lewis' simulation method for point processes*. IEEE Trans. Inf. Theory.
- Ozaki, T. (1979). *Maximum likelihood estimation of Hawkes' self-exciting point processes*. Ann. Inst. Stat. Math.
- Bacry, E., Mastromatteo, I., Muzy, J.-F. (2015). *Hawkes processes in finance*. Market Microstructure and Liquidity.
