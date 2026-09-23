"""Core simulations for the Edge Dynamics Lab Streamlit application.

The functions in this module are deliberately independent of Streamlit so they
can be imported into notebooks, scripts, or tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, signal, stats

EPS = np.finfo(float).eps


def zscore(x: ArrayLike, axis: int = 0) -> NDArray[np.float64]:
    """Z-score an array with population standard deviation (ddof=0)."""
    arr = np.asarray(x, dtype=float)
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, ddof=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (arr - mean) / std


def upper_triangle_values(matrix: ArrayLike, k: int = 1) -> NDArray[np.float64]:
    """Return upper-triangular entries as a vector."""
    mat = np.asarray(matrix, dtype=float)
    return mat[np.triu_indices_from(mat, k=k)]


def matrix_edge_correlation(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation between off-diagonal upper triangles."""
    va = upper_triangle_values(a)
    vb = upper_triangle_values(b)
    if np.std(va) < EPS or np.std(vb) < EPS:
        return float("nan")
    return float(np.corrcoef(va, vb)[0, 1])


def autocorrelation(x: ArrayLike, max_lag: int) -> NDArray[np.float64]:
    """Biased sample autocorrelation from lag 0 through max_lag."""
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr - np.mean(arr)
    denom = float(np.dot(arr, arr))
    if denom < EPS:
        out = np.zeros(max_lag + 1, dtype=float)
        out[0] = 1.0
        return out
    corr = signal.correlate(arr, arr, mode="full", method="fft")
    mid = len(arr) - 1
    return corr[mid : mid + max_lag + 1] / denom


def welch_psd(
    x: ArrayLike,
    tr: float,
    nperseg: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One-sided Welch power spectral density."""
    arr = np.asarray(x, dtype=float).ravel()
    if nperseg is None:
        nperseg = min(len(arr), 512)
    nperseg = max(16, min(int(nperseg), len(arr)))
    f, pxx = signal.welch(
        arr,
        fs=1.0 / tr,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )
    return f, pxx


def cospectrum_and_cumulative(
    x: ArrayLike,
    y: ArrayLike,
    tr: float,
    nperseg: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Estimate co-spectrum and cumulative signed covariance contribution.

    SciPy's one-sided density convention approximately integrates to the
    zero-lag covariance for real-valued signals.
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if len(xa) != len(ya):
        raise ValueError("x and y must have the same length")
    if nperseg is None:
        nperseg = min(len(xa), 512)
    nperseg = max(16, min(int(nperseg), len(xa)))
    f, pxy = signal.csd(
        xa,
        ya,
        fs=1.0 / tr,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )
    cospec = np.real(pxy)
    cumulative = integrate.cumulative_trapezoid(cospec, f, initial=0.0)
    total = float(cumulative[-1])
    if abs(total) < 1e-12:
        fraction = np.full_like(cumulative, np.nan)
    else:
        fraction = cumulative / total
    return f, cospec, fraction


def fraction_at_frequency(
    frequencies: ArrayLike,
    cumulative_fraction: ArrayLike,
    target_hz: float,
) -> float:
    """Interpolate a cumulative spectral fraction at target_hz."""
    f = np.asarray(frequencies, dtype=float)
    c = np.asarray(cumulative_fraction, dtype=float)
    if len(f) == 0 or np.all(~np.isfinite(c)):
        return float("nan")
    target = float(np.clip(target_hz, f[0], f[-1]))
    return float(np.interp(target, f, c))


def lowpass_correlated_pair(
    n_time: int,
    tr: float,
    rho: float,
    cutoff_hz: float,
    order: int = 4,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate two approximately band-limited signals with exact sample rho.

    Independent low-pass Gaussian signals are orthogonalized after filtering;
    the second output is then constructed as rho*x + sqrt(1-rho^2)*u.
    """
    if n_time < 64:
        raise ValueError("n_time must be at least 64")
    fs = 1.0 / tr
    nyquist = fs / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise ValueError("cutoff_hz must lie between 0 and the Nyquist frequency")
    if not -0.999 < rho < 0.999:
        raise ValueError("rho must lie between -0.999 and 0.999")

    rng = np.random.default_rng(seed)
    sos = signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, rng.normal(size=n_time))
    u = signal.sosfiltfilt(sos, rng.normal(size=n_time))
    x = zscore(x)
    u = u - x * (np.dot(u, x) / np.dot(x, x))
    u = zscore(u)
    y = rho * x + np.sqrt(1.0 - rho**2) * u
    y = zscore(y)
    return x, y


def frequency_mixing_signals(
    duration_s: float,
    tr: float,
    f1_hz: float,
    f2_hz: float,
    phase_deg: float,
    noise_sd: float = 0.0,
    seed: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """Generate sinusoids, their real product, and analytic conjugate product."""
    n_time = int(np.floor(duration_s / tr))
    if n_time < 32:
        raise ValueError("duration/tr must yield at least 32 samples")
    t = np.arange(n_time, dtype=float) * tr
    phase = np.deg2rad(phase_deg)
    x = np.cos(2.0 * np.pi * f1_hz * t)
    y = np.cos(2.0 * np.pi * f2_hz * t + phase)
    if noise_sd > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(scale=noise_sd, size=n_time)
        y = y + rng.normal(scale=noise_sd, size=n_time)
    real_edge = x * y
    ax = signal.hilbert(x)
    ay = signal.hilbert(y)
    analytic_edge = 0.5 * np.real(ax * np.conj(ay))
    return {
        "time": t,
        "x": x,
        "y": y,
        "real_edge": real_edge,
        "analytic_edge": analytic_edge,
    }


def fft_amplitude(x: ArrayLike, tr: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return one-sided FFT amplitude with the DC term retained."""
    arr = np.asarray(x, dtype=float).ravel()
    n = len(arr)
    spectrum = np.fft.rfft(arr)
    amplitude = np.abs(spectrum) / n
    if n > 1:
        amplitude[1:-1] *= 2.0
    f = np.fft.rfftfreq(n, d=tr)
    return f, amplitude


def ar1_matrix(
    n_time: int,
    n_series: int,
    phi: float,
    rng: np.random.Generator,
    innovation_cov: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Simulate a stationary multivariate AR(1) with common scalar phi."""
    if not -0.999 < phi < 0.999:
        raise ValueError("phi must lie between -0.999 and 0.999")
    if innovation_cov is None:
        cov = np.eye(n_series)
    else:
        cov = np.asarray(innovation_cov, dtype=float)
        if cov.shape != (n_series, n_series):
            raise ValueError("innovation_cov has the wrong shape")
    chol = np.linalg.cholesky(cov)
    out = np.empty((n_time, n_series), dtype=float)
    out[0] = rng.normal(size=n_series) @ chol.T
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, n_time):
        eps = rng.normal(size=n_series) @ chol.T
        out[t] = phi * out[t - 1] + scale * eps
    return out


def ar1_correlated_pair(
    n_time: int,
    phi: float,
    rho: float,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a bivariate AR(1) with stationary correlation rho."""
    if not -0.999 < rho < 0.999:
        raise ValueError("rho must lie between -0.999 and 0.999")
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    x = ar1_matrix(n_time, 2, phi, rng, innovation_cov=cov)
    return zscore(x[:, 0]), zscore(x[:, 1])


def contiguous_run_lengths(mask: ArrayLike) -> NDArray[np.int64]:
    """Lengths of True runs in a boolean array."""
    arr = np.asarray(mask, dtype=bool).ravel()
    if len(arr) == 0 or not np.any(arr):
        return np.array([], dtype=np.int64)
    padded = np.r_[False, arr, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(np.int64)


@dataclass(frozen=True)
class ModularSimulation:
    data: NDArray[np.float64]
    population_correlation: NDArray[np.float64]
    labels: NDArray[np.int64]


def modular_factor_process(
    n_time: int,
    n_modules: int,
    nodes_per_module: int,
    within_corr: float,
    between_corr: float,
    phi: float = 0.0,
    seed: int = 0,
) -> ModularSimulation:
    """Generate a modular process from global, module, and private factors.

    For distinct nodes in the same module, population correlation is
    ``within_corr``. Across modules it is ``between_corr``. Nonnegative values
    satisfying 0 <= between_corr <= within_corr < 1 are guaranteed PSD.
    """
    if n_modules < 1 or nodes_per_module < 2:
        raise ValueError("Need at least one module and two nodes per module")
    if not 0 <= between_corr <= within_corr < 1:
        raise ValueError("Require 0 <= between_corr <= within_corr < 1")

    rng = np.random.default_rng(seed)
    n_nodes = n_modules * nodes_per_module
    labels = np.repeat(np.arange(n_modules), nodes_per_module)

    global_factor = ar1_matrix(n_time, 1, phi, rng)[:, 0]
    module_factors = ar1_matrix(n_time, n_modules, phi, rng)
    private_factors = ar1_matrix(n_time, n_nodes, phi, rng)

    g_load = np.sqrt(between_corr)
    m_load = np.sqrt(within_corr - between_corr)
    u_load = np.sqrt(1.0 - within_corr)

    data = np.empty((n_time, n_nodes), dtype=float)
    for node in range(n_nodes):
        module = labels[node]
        data[:, node] = (
            g_load * global_factor
            + m_load * module_factors[:, module]
            + u_load * private_factors[:, node]
        )
    data = zscore(data, axis=0)

    population = np.full((n_nodes, n_nodes), between_corr, dtype=float)
    for module in range(n_modules):
        idx = np.flatnonzero(labels == module)
        population[np.ix_(idx, idx)] = within_corr
    np.fill_diagonal(population, 1.0)
    return ModularSimulation(data=data, population_correlation=population, labels=labels)


def edge_rss(data: ArrayLike) -> NDArray[np.float64]:
    """Whole-brain upper-triangle edge root-sum-square at every time point."""
    z = np.asarray(data, dtype=float)
    q2 = np.sum(z**2, axis=1)
    q4 = np.sum(z**4, axis=1)
    rss2 = 0.5 * (q2**2 - q4)
    return np.sqrt(np.maximum(rss2, 0.0))


def edge_rss_from_explicit_edges(data: ArrayLike) -> NDArray[np.float64]:
    """Reference implementation of edge RSS using explicit edge products."""
    z = np.asarray(data, dtype=float)
    i, j = np.triu_indices(z.shape[1], k=1)
    edges = z[:, i] * z[:, j]
    return np.sqrt(np.sum(edges**2, axis=1))


def mean_outer_product(data: ArrayLike, indices: Iterable[int] | NDArray[np.int64]) -> NDArray[np.float64]:
    """Mean framewise outer product over selected indices."""
    z = np.asarray(data, dtype=float)
    idx = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices, dtype=int)
    if idx.size == 0:
        raise ValueError("At least one index is required")
    selected = z[idx]
    return selected.T @ selected / idx.size


@dataclass(frozen=True)
class EventReconstruction:
    rss: NDArray[np.float64]
    top_indices: NDArray[np.int64]
    bottom_indices: NDArray[np.int64]
    random_indices: NDArray[np.int64]
    full_pattern: NDArray[np.float64]
    top_pattern: NDArray[np.float64]
    bottom_pattern: NDArray[np.float64]
    random_pattern: NDArray[np.float64]
    full_score: float
    top_score: float
    bottom_score: float
    random_score: float
    random_score_distribution: NDArray[np.float64]


def event_reconstruction(
    data: ArrayLike,
    target_matrix: ArrayLike,
    top_fraction: float = 0.05,
    n_random: int = 200,
    seed: int = 0,
) -> EventReconstruction:
    """Compare edge-pattern reconstruction from high-, low-, and random-RSS frames."""
    z = np.asarray(data, dtype=float)
    target = np.asarray(target_matrix, dtype=float)
    n_time = z.shape[0]
    n_select = max(2, int(round(n_time * top_fraction)))
    n_select = min(n_select, n_time // 2)

    rss = edge_rss(z)
    order = np.argsort(rss)
    bottom = order[:n_select]
    top = order[-n_select:]

    rng = np.random.default_rng(seed)
    random_idx = np.sort(rng.choice(n_time, size=n_select, replace=False))

    full_pattern = mean_outer_product(z, np.arange(n_time))
    top_pattern = mean_outer_product(z, top)
    bottom_pattern = mean_outer_product(z, bottom)
    random_pattern = mean_outer_product(z, random_idx)

    random_scores = np.empty(n_random, dtype=float)
    for rep in range(n_random):
        idx = rng.choice(n_time, size=n_select, replace=False)
        random_scores[rep] = matrix_edge_correlation(mean_outer_product(z, idx), target)

    return EventReconstruction(
        rss=rss,
        top_indices=top,
        bottom_indices=bottom,
        random_indices=random_idx,
        full_pattern=full_pattern,
        top_pattern=top_pattern,
        bottom_pattern=bottom_pattern,
        random_pattern=random_pattern,
        full_score=matrix_edge_correlation(full_pattern, target),
        top_score=matrix_edge_correlation(top_pattern, target),
        bottom_score=matrix_edge_correlation(bottom_pattern, target),
        random_score=matrix_edge_correlation(random_pattern, target),
        random_score_distribution=random_scores,
    )


@dataclass(frozen=True)
class CAPResult:
    cap: NDArray[np.float64]
    theoretical_cap: NDArray[np.float64]
    selected: NDArray[np.bool_]
    sample_fc_seed: NDArray[np.float64]
    population_fc_seed: NDArray[np.float64]
    inverse_mills_ratio: float


def gaussian_cap(
    data: ArrayLike,
    population_correlation: ArrayLike,
    seed_node: int,
    threshold_z: float,
) -> CAPResult:
    """Threshold a seed and compare its CAP with the Gaussian conditional mean."""
    z = np.asarray(data, dtype=float)
    population = np.asarray(population_correlation, dtype=float)
    if not 0 <= seed_node < z.shape[1]:
        raise ValueError("seed_node is out of range")
    selected = z[:, seed_node] > threshold_z
    if np.count_nonzero(selected) < 2:
        raise ValueError("Threshold selected fewer than two frames")
    cap = np.mean(z[selected], axis=0)
    inverse_mills = float(stats.norm.pdf(threshold_z) / stats.norm.sf(threshold_z))
    theoretical = population[:, seed_node] * inverse_mills
    sample_fc = np.corrcoef(z, rowvar=False)[:, seed_node]
    return CAPResult(
        cap=cap,
        theoretical_cap=theoretical,
        selected=selected,
        sample_fc_seed=sample_fc,
        population_fc_seed=population[:, seed_node],
        inverse_mills_ratio=inverse_mills,
    )


def canonical_hrf(
    tr: float,
    duration_s: float = 32.0,
    width_scale: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A normalized double-gamma HRF for demonstration purposes."""
    if tr <= 0 or width_scale <= 0:
        raise ValueError("tr and width_scale must be positive")
    t = np.arange(0.0, duration_s + tr, tr)
    peak = stats.gamma.pdf(t, a=6.0, scale=width_scale)
    undershoot = stats.gamma.pdf(t, a=16.0, scale=width_scale) / 6.0
    h = peak - undershoot
    norm = np.max(np.abs(h))
    if norm > 0:
        h = h / norm
    return t, h


def _compound_bernoulli_events(
    n_time: int,
    probability: float,
    rng: np.random.Generator,
    amplitude_sigma: float = 0.35,
) -> NDArray[np.float64]:
    """Sparse positive impulses with mean-one lognormal amplitudes."""
    probability = float(np.clip(probability, 0.0, 0.95))
    mask = rng.random(n_time) < probability
    amplitudes = rng.lognormal(
        mean=-0.5 * amplitude_sigma**2,
        sigma=amplitude_sigma,
        size=n_time,
    )
    return mask.astype(float) * amplitudes


@dataclass(frozen=True)
class EventDrivenBOLD:
    time: NDArray[np.float64]
    neural_1: NDArray[np.float64]
    neural_2: NDArray[np.float64]
    bold_1: NDArray[np.float64]
    bold_2: NDArray[np.float64]
    hrf_time: NDArray[np.float64]
    hrf: NDArray[np.float64]
    shared_events: NDArray[np.float64]


def event_driven_bold(
    duration_s: float,
    tr: float,
    events_per_minute: float,
    shared_fraction: float,
    noise_sd: float,
    hrf_width: float,
    seed: int = 0,
) -> EventDrivenBOLD:
    """Generate sparse shared/private events and convolve them into BOLD-like signals."""
    n_time = int(np.floor(duration_s / tr))
    if n_time < 64:
        raise ValueError("duration/tr must yield at least 64 samples")
    if not 0 <= shared_fraction <= 1:
        raise ValueError("shared_fraction must be in [0, 1]")
    if events_per_minute < 0:
        raise ValueError("events_per_minute must be nonnegative")

    rng = np.random.default_rng(seed)
    total_rate_hz = events_per_minute / 60.0
    p_total = total_rate_hz * tr
    p_shared = p_total * shared_fraction
    p_private = p_total * (1.0 - shared_fraction)

    shared = _compound_bernoulli_events(n_time, p_shared, rng)
    private_1 = _compound_bernoulli_events(n_time, p_private, rng)
    private_2 = _compound_bernoulli_events(n_time, p_private, rng)

    neural_1 = shared + private_1
    neural_2 = shared + private_2

    hrf_time, hrf = canonical_hrf(tr, width_scale=hrf_width)
    bold_1 = signal.fftconvolve(neural_1, hrf, mode="full")[:n_time]
    bold_2 = signal.fftconvolve(neural_2, hrf, mode="full")[:n_time]

    if noise_sd > 0:
        reference = 0.5 * (np.std(bold_1) + np.std(bold_2))
        scale = noise_sd * max(reference, 1e-6)
        bold_1 = bold_1 + rng.normal(scale=scale, size=n_time)
        bold_2 = bold_2 + rng.normal(scale=scale, size=n_time)

    bold_1 = zscore(bold_1)
    bold_2 = zscore(bold_2)
    time = np.arange(n_time, dtype=float) * tr
    return EventDrivenBOLD(
        time=time,
        neural_1=neural_1,
        neural_2=neural_2,
        bold_1=bold_1,
        bold_2=bold_2,
        hrf_time=hrf_time,
        hrf=hrf,
        shared_events=shared,
    )
