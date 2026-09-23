# Edge Dynamics Lab

A reproducible interactive Python laboratory for exploring how low-frequency BOLD covariance, autocorrelation, edge time series, cofluctuation events, co-activation patterns, and sparse point processes can coexist.

The main interface is a Streamlit app with six simulations. An optional Jupyter/ipywidgets notebook provides editable versions of the principal demonstrations. The numerical model code is kept in `edge_lab.py`, independent of either interface, so it can also be imported into analysis scripts.

## What the app demonstrates

1. **Slow nodes → bursty edges**  
   Two correlated low-pass signals have covariance concentrated at low frequencies, while their instantaneous product is irregular and heavy-tailed.

2. **Frequency mixing**  
   Multiplying two real-valued oscillations creates difference- and sum-frequency components. Equal-frequency signals yield a DC term plus a component at twice the carrier frequency. An analytic conjugate product is shown for comparison.

3. **Autocorrelation**  
   For a symmetric bivariate AR(1), the nodal ACF is approximately `phi^k`, while the centered edge-product ACF is approximately `phi^(2k)`.

4. **Static covariance events**  
   A stationary modular Gaussian factor process—with no state switching—generates high-RSS frames, event episodes, recognizable network topographies, and unequal framewise leverage on covariance estimation.

5. **Gaussian CAPs**  
   Thresholding a seed produces a CAP proportional to the seed correlation map under a static multivariate Gaussian model:

   ```text
   E[z | z_seed > u] = R[:, seed] * phi_normal(u) / survival_normal(u)
   ```

6. **Sparse events → slow BOLD**  
   Shared and private point events are convolved with a double-gamma hemodynamic kernel. The latent process is punctate, while observed BOLD covariance is smooth and concentrated at low frequency.

## Installation

Python 3.10 or newer is recommended.

### macOS or Linux

```bash
cd edge_dynamics_lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

You can also run:

```bash
./run_app.sh
```

### Windows PowerShell

```powershell
cd edge_dynamics_lab
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Or double-click `run_app.bat` after activating the environment and installing the requirements.

Streamlit will print a local URL, normally `http://localhost:8501`.


## Notebook alternative

Install the notebook environment from the project directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-notebook.txt
python -m jupyter lab notebooks/edge_dynamics_widgets.ipynb
```

You can also run `./run_notebook.sh` on macOS/Linux or `run_notebook.bat` on Windows after installing the notebook requirements.

The notebook contains widgets for low-frequency nodes and edge products, frequency mixing, AR(1) edge autocorrelation, stationary-covariance events, and sparse-event hemodynamic filtering.

## Run the tests

From the project directory:

```bash
python -m pytest -q
```

The tests verify:

- requested sample correlation and the edge-mean identity;
- DC and double-frequency terms for equal-frequency multiplication;
- the squared-ACF prediction for AR(1) edge products;
- equivalence of explicit edge RSS and its closed-form nodal expression;
- the Gaussian CAP conditional-mean prediction;
- output shape and standardization of the event-driven BOLD simulation;
- execution of all default Streamlit app branches through a dependency-light smoke test.

## Reuse the simulation functions

```python
from edge_lab import lowpass_correlated_pair, edge_rss

x, y = lowpass_correlated_pair(
    n_time=2000,
    tr=0.72,
    rho=0.4,
    cutoff_hz=0.08,
    seed=7,
)

edge = x * y
print(edge.mean())  # approximately—and by construction here, exactly—0.4
```

The principal reusable functions are:

- `lowpass_correlated_pair`
- `frequency_mixing_signals`
- `ar1_correlated_pair`
- `modular_factor_process`
- `edge_rss`
- `event_reconstruction`
- `gaussian_cap`
- `event_driven_bold`
- `cospectrum_and_cumulative`

## Suggested experiments

- In tab 1, hold the low-pass cutoff fixed and increase the event threshold. The point set becomes sparse without changing the generating process.
- In tab 2, set `f1 = f2 = 0.05 Hz`. The real edge contains DC and `0.10 Hz`, not `0.05 Hz`.
- In tab 3, increase the AR coefficient. Events become temporally clustered, but the edge ACF remains faster than the node ACF.
- In tab 4, set temporal AR to zero. High-RSS frames remain, showing that autocorrelation is not necessary for extreme covariance contributions.
- In tab 5, raise the seed threshold. Fewer frames are selected, but the expected CAP direction remains the seed correlation map.
- In tab 6, eliminate shared events. Nodal BOLD remains low-frequency, but cross-covariance collapses toward zero.

## Scope and limitations

This is explanatory simulation software, not a validated fMRI preprocessing or inference package. The models omit many empirical complexities, including spatially heterogeneous HRFs, censoring, global and regional physiology, motion, sampling irregularities, non-Gaussian innovations, and uncertainty in covariance estimation.

For empirical work, compare event statistics with nulls that preserve the relevant lower-order structure—ideally the full cross-spectrum, regional marginal distributions, censoring structure, and measured physiology—and use held-out data where possible.
