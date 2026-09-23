from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import integrate, stats

from edge_lab import (
    ar1_correlated_pair,
    autocorrelation,
    contiguous_run_lengths,
    cospectrum_and_cumulative,
    edge_rss,
    event_driven_bold,
    event_reconstruction,
    fft_amplitude,
    fraction_at_frequency,
    frequency_mixing_signals,
    gaussian_cap,
    lowpass_correlated_pair,
    matrix_edge_correlation,
    modular_factor_process,
    welch_psd,
)


st.set_page_config(
    page_title="Edge Dynamics Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def finish_figure(fig: go.Figure, height: int = 430, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        margin=dict(l=55, r=25, t=60 if title else 30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def matrix_heatmap(matrix: np.ndarray, title: str) -> go.Figure:
    mat = np.asarray(matrix, dtype=float).copy()
    offdiag = mat[~np.eye(mat.shape[0], dtype=bool)]
    finite = offdiag[np.isfinite(offdiag)]
    bound = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    bound = max(bound, 1e-6)
    fig = go.Figure(
        go.Heatmap(
            z=mat,
            zmid=0,
            zmin=-bound,
            zmax=bound,
            colorbar=dict(title="value"),
            hovertemplate="node i=%{y}<br>node j=%{x}<br>value=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return finish_figure(fig, height=410, title=title)




def format_spectral_fraction(value: float, correlation: float) -> str:
    if not np.isfinite(value) or abs(correlation) < 0.05:
        return "n/a (r≈0)"
    return f"{value * 100:.1f}%"

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def event_window(signal_values: np.ndarray, tr: float, window_s: float) -> slice:
    n = len(signal_values)
    width = min(n, max(20, int(round(window_s / tr))))
    center = int(np.argmax(np.abs(signal_values)))
    start = int(np.clip(center - width // 2, 0, max(0, n - width)))
    return slice(start, start + width)


def download_csv(label: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


@st.cache_data(show_spinner=False)
def cached_lowpass_pair(n_time: int, tr: float, rho: float, cutoff_hz: float, seed: int):
    return lowpass_correlated_pair(n_time, tr, rho, cutoff_hz, seed=seed)


@st.cache_data(show_spinner=False)
def cached_frequency_mixing(
    duration_s: float, tr: float, f1_hz: float, f2_hz: float, phase_deg: float, noise_sd: float, seed: int
):
    return frequency_mixing_signals(duration_s, tr, f1_hz, f2_hz, phase_deg, noise_sd, seed)


@st.cache_data(show_spinner=False)
def cached_ar1_pair(n_time: int, phi: float, rho: float, seed: int):
    return ar1_correlated_pair(n_time, phi, rho, seed)


@st.cache_data(show_spinner=False)
def cached_static_event_simulation(
    n_time: int,
    n_modules: int,
    nodes_per_module: int,
    within_corr: float,
    between_corr: float,
    phi: float,
    top_fraction: float,
    seed: int,
):
    sim = modular_factor_process(
        n_time=n_time,
        n_modules=n_modules,
        nodes_per_module=nodes_per_module,
        within_corr=within_corr,
        between_corr=between_corr,
        phi=phi,
        seed=seed,
    )
    recon = event_reconstruction(
        sim.data,
        sim.population_correlation,
        top_fraction=top_fraction,
        n_random=300,
        seed=seed + 360,
    )
    return sim, recon


@st.cache_data(show_spinner=False)
def cached_cap_simulation(
    n_time: int, n_modules: int, nodes_per_module: int, phi: float, seed_node: int, threshold_z: float, seed: int
):
    sim = modular_factor_process(
        n_time=n_time,
        n_modules=n_modules,
        nodes_per_module=nodes_per_module,
        within_corr=0.50,
        between_corr=0.08,
        phi=phi,
        seed=seed,
    )
    cap = gaussian_cap(sim.data, sim.population_correlation, seed_node, threshold_z)
    return sim, cap


@st.cache_data(show_spinner=False)
def cached_event_bold(
    duration_s: float,
    tr: float,
    events_per_minute: float,
    shared_fraction: float,
    noise_sd: float,
    hrf_width: float,
    seed: int,
):
    return event_driven_bold(
        duration_s=duration_s,
        tr=tr,
        events_per_minute=events_per_minute,
        shared_fraction=shared_fraction,
        noise_sd=noise_sd,
        hrf_width=hrf_width,
        seed=seed,
    )


st.title("Edge Dynamics Lab")
st.caption(
    "Interactive simulations connecting low-frequency BOLD covariance, autocorrelation, "
    "edge time series, cofluctuation events, CAPs, and sparse latent point processes."
)

with st.sidebar:
    st.header("Global controls")
    global_seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=7, step=1)
    st.markdown(
        "Each panel is a separate simulation using this seed. Change one parameter at a time "
        "to see which phenomena are algebraic, which depend on autocorrelation, and which require "
        "a genuinely time-varying process."
    )
    st.divider()
    st.markdown(
        "**Interpretation rule:** an edge value $z_i(t)z_j(t)$ is a framewise contribution to "
        "covariance, not a one-sample estimate of a bounded correlation parameter."
    )


tabs = st.tabs(
    [
        "1 · Slow nodes → bursty edges",
        "2 · Frequency mixing",
        "3 · Autocorrelation",
        "4 · Static covariance events",
        "5 · Gaussian CAPs",
        "6 · Sparse events → slow BOLD",
        "Guide",
    ]
)


with tabs[0]:
    st.subheader("Low-frequency node signals can have intermittent-looking edge products")
    st.markdown(
        r"The static correlation is the temporal mean, "
        r"$\widehat r_{ij}=T^{-1}\sum_t z_i(t)z_j(t)$. "
        r"Low-frequency cross-spectral mass constrains this mean, but multiplication changes the spectrum "
        r"and produces a heavy-tailed product process."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    duration_1 = c1.slider("Duration (s)", 240, 1800, 900, 60, key="d1")
    tr_1 = c2.slider("TR (s)", 0.40, 2.00, 0.72, 0.04, key="tr1")
    cutoff_1 = c3.slider("Low-pass cutoff (Hz)", 0.01, 0.20, 0.08, 0.005, key="cut1")
    rho_1 = c4.slider("Target sample correlation", -0.80, 0.80, 0.40, 0.05, key="rho1")
    pct_1 = c5.slider("Event percentile |edge|", 80, 99, 95, 1, key="pct1")

    n_time_1 = int(duration_1 / tr_1)
    x1, y1 = cached_lowpass_pair(
        n_time=n_time_1,
        tr=tr_1,
        rho=rho_1,
        cutoff_hz=min(cutoff_1, 0.95 * (0.5 / tr_1)),
        seed=int(global_seed) + 11,
    )
    edge1 = x1 * y1
    threshold1 = float(np.quantile(np.abs(edge1), pct_1 / 100.0))
    events1 = np.abs(edge1) >= threshold1
    t1 = np.arange(n_time_1) * tr_1
    view1 = event_window(edge1, tr_1, window_s=150.0)

    f_x1, p_x1 = welch_psd(x1, tr_1)
    f_y1, p_y1 = welch_psd(y1, tr_1)
    f_e1, p_e1 = welch_psd(edge1 - np.mean(edge1), tr_1)
    f_c1, cospec1, cumulative1 = cospectrum_and_cumulative(x1, y1, tr_1)
    below_01 = fraction_at_frequency(f_c1, cumulative1, 0.10)
    below_02 = fraction_at_frequency(f_c1, cumulative1, 0.20)
    sample_r1 = float(np.corrcoef(x1, y1)[0, 1])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sample correlation", f"{sample_r1:.3f}")
    m2.metric("Frames above threshold", f"{events1.mean() * 100:.1f}%")
    m3.metric("Signed covariance ≤ 0.10 Hz", format_spectral_fraction(below_01, sample_r1))
    m4.metric("Signed covariance ≤ 0.20 Hz", format_spectral_fraction(below_02, sample_r1))

    fig_ts1 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Band-limited node signals", "Their instantaneous product"),
    )
    fig_ts1.add_trace(go.Scatter(x=t1[view1], y=x1[view1], name="node x", mode="lines"), row=1, col=1)
    fig_ts1.add_trace(go.Scatter(x=t1[view1], y=y1[view1], name="node y", mode="lines"), row=1, col=1)
    fig_ts1.add_trace(go.Scatter(x=t1[view1], y=edge1[view1], name="edge x·y", mode="lines"), row=2, col=1)
    local_events = np.flatnonzero(events1[view1])
    if local_events.size:
        local_t = t1[view1][local_events]
        local_y = edge1[view1][local_events]
        fig_ts1.add_trace(
            go.Scatter(x=local_t, y=local_y, name="threshold events", mode="markers", marker=dict(size=7)),
            row=2,
            col=1,
        )
    fig_ts1.add_hline(y=threshold1, line_dash="dot", row=2, col=1)
    fig_ts1.add_hline(y=-threshold1, line_dash="dot", row=2, col=1)
    fig_ts1.update_xaxes(title_text="time (s)", row=2, col=1)
    fig_ts1.update_yaxes(title_text="z", row=1, col=1)
    fig_ts1.update_yaxes(title_text="product", row=2, col=1)
    st.plotly_chart(finish_figure(fig_ts1, height=600), use_container_width=True, config=PLOT_CONFIG)

    left, right = st.columns(2)
    with left:
        fig_psd1 = go.Figure()
        fig_psd1.add_trace(go.Scatter(x=f_x1, y=p_x1 / integrate.trapezoid(p_x1, f_x1), name="node x PSD"))
        fig_psd1.add_trace(go.Scatter(x=f_y1, y=p_y1 / integrate.trapezoid(p_y1, f_y1), name="node y PSD"))
        fig_psd1.add_trace(go.Scatter(x=f_e1, y=p_e1 / integrate.trapezoid(p_e1, f_e1), name="centered edge PSD"))
        fig_psd1.add_vline(x=cutoff_1, line_dash="dot", annotation_text="node cutoff")
        fig_psd1.update_xaxes(title="frequency (Hz)", range=[0, min(0.4, 0.5 / tr_1)])
        fig_psd1.update_yaxes(title="area-normalized PSD")
        st.plotly_chart(
            finish_figure(fig_psd1, title="Multiplication broadens and mixes frequency content"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        fig_cos1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
        fig_cos1.add_trace(go.Scatter(x=f_c1, y=cospec1, name="co-spectrum"), row=1, col=1)
        fig_cos1.add_trace(go.Scatter(x=f_c1, y=cumulative1, name="cumulative fraction"), row=2, col=1)
        fig_cos1.add_hline(y=1.0, line_dash="dot", row=2, col=1)
        fig_cos1.add_vline(x=0.10, line_dash="dot", row=2, col=1)
        fig_cos1.add_vline(x=0.20, line_dash="dot", row=2, col=1)
        fig_cos1.update_xaxes(title_text="frequency (Hz)", range=[0, min(0.4, 0.5 / tr_1)], row=2, col=1)
        fig_cos1.update_yaxes(title_text="co-spectrum", row=1, col=1)
        fig_cos1.update_yaxes(title_text="fraction of signed covariance", row=2, col=1)
        st.plotly_chart(
            finish_figure(fig_cos1, title="Correlation as integrated co-spectrum"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )

    with st.expander("Inspect the edge-value distribution and export this simulation"):
        colh, cold = st.columns([2, 1])
        with colh:
            fig_hist1 = go.Figure(go.Histogram(x=edge1, nbinsx=90, histnorm="probability density"))
            fig_hist1.add_vline(x=np.mean(edge1), line_dash="dot", annotation_text="mean = correlation")
            fig_hist1.update_xaxes(title="edge value zᵢzⱼ")
            fig_hist1.update_yaxes(title="density")
            st.plotly_chart(finish_figure(fig_hist1, height=360), use_container_width=True, config=PLOT_CONFIG)
        with cold:
            st.markdown(
                "Most products are modest, while coincident large amplitudes generate long-tailed excursions. "
                "Those excursions can dominate a finite covariance sum without implying that a latent coupling "
                "parameter switched on only at those frames."
            )
            download_csv(
                "Download node and edge time series",
                pd.DataFrame({"time_s": t1, "x": x1, "y": y1, "edge": edge1, "event": events1}),
                "slow_nodes_bursty_edges.csv",
                "download_tab1",
            )


with tabs[1]:
    st.subheader("The ordinary real-valued edge product is a frequency mixer")
    st.latex(
        r"\cos(2\pi f_1t)\cos(2\pi f_2t+\phi)="
        r"\tfrac12\cos(2\pi(f_1-f_2)t-\phi)+\tfrac12\cos(2\pi(f_1+f_2)t+\phi)"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    f1_2 = c1.slider("f₁ (Hz)", 0.005, 0.200, 0.050, 0.005, key="f12")
    f2_2 = c2.slider("f₂ (Hz)", 0.005, 0.200, 0.050, 0.005, key="f22")
    phase_2 = c3.slider("Phase offset (°)", -180, 180, 45, 5, key="phase2")
    tr_2 = c4.slider("Sample interval (s)", 0.20, 1.50, 0.50, 0.05, key="tr2")
    duration_2 = c5.slider("Duration (s)", 120, 1200, 400, 20, key="duration2")
    noise_2 = c6.slider("Additive noise SD", 0.0, 1.0, 0.0, 0.05, key="noise2")

    mix2 = cached_frequency_mixing(
        duration_s=duration_2,
        tr=tr_2,
        f1_hz=f1_2,
        f2_hz=f2_2,
        phase_deg=phase_2,
        noise_sd=noise_2,
        seed=int(global_seed) + 22,
    )
    t2 = mix2["time"]
    show2 = t2 <= min(160.0, duration_2)

    diff_f = abs(f1_2 - f2_2)
    sum_f = f1_2 + f2_2
    nyquist2 = 0.5 / tr_2
    aliased_sum = abs(((sum_f + nyquist2) % (2 * nyquist2)) - nyquist2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Difference component", f"{diff_f:.3f} Hz")
    m2.metric("Sum component", f"{sum_f:.3f} Hz")
    m3.metric("Observed sum after aliasing", f"{aliased_sum:.3f} Hz")
    m4.metric("Mean real edge", f"{np.mean(mix2['real_edge']):.3f}")

    fig_mix2 = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Node signals", "Ordinary real product", "Analytic conjugate product"),
    )
    fig_mix2.add_trace(go.Scatter(x=t2[show2], y=mix2["x"][show2], name="x"), row=1, col=1)
    fig_mix2.add_trace(go.Scatter(x=t2[show2], y=mix2["y"][show2], name="y"), row=1, col=1)
    fig_mix2.add_trace(go.Scatter(x=t2[show2], y=mix2["real_edge"][show2], name="x·y"), row=2, col=1)
    fig_mix2.add_trace(
        go.Scatter(x=t2[show2], y=mix2["analytic_edge"][show2], name="½ Re[aₓaᵧ*]"), row=3, col=1
    )
    fig_mix2.update_xaxes(title_text="time (s)", row=3, col=1)
    st.plotly_chart(finish_figure(fig_mix2, height=720), use_container_width=True, config=PLOT_CONFIG)

    f_x2, a_x2 = fft_amplitude(mix2["x"], tr_2)
    f_y2, a_y2 = fft_amplitude(mix2["y"], tr_2)
    f_r2, a_r2 = fft_amplitude(mix2["real_edge"], tr_2)
    f_a2, a_a2 = fft_amplitude(mix2["analytic_edge"], tr_2)
    fig_fft2 = go.Figure()
    fig_fft2.add_trace(go.Scatter(x=f_x2, y=a_x2, name="node x"))
    fig_fft2.add_trace(go.Scatter(x=f_y2, y=a_y2, name="node y"))
    fig_fft2.add_trace(go.Scatter(x=f_r2, y=a_r2, name="real edge"))
    fig_fft2.add_trace(go.Scatter(x=f_a2, y=a_a2, name="analytic edge"))
    fig_fft2.add_vline(x=diff_f, line_dash="dot", annotation_text="|f₁−f₂|")
    fig_fft2.add_vline(x=aliased_sum, line_dash="dash", annotation_text="f₁+f₂ (observed)")
    fig_fft2.update_xaxes(title="frequency (Hz)", range=[0, nyquist2])
    fig_fft2.update_yaxes(title="FFT amplitude")
    st.plotly_chart(
        finish_figure(fig_fft2, title="The edge spectrum need not contain either original carrier frequency"),
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    st.info(
        "Set f₁=f₂. The real product becomes a DC term plus a 2f oscillation; the analytic conjugate "
        "product becomes approximately constant for fixed phase. This is why looking for the node carrier "
        "frequency inside an ordinary edge time series can be misleading."
    )


with tabs[2]:
    st.subheader("Edge autocorrelation can decay faster than nodal autocorrelation")
    st.latex(
        r"\operatorname{Cov}[e_{ij}(t),e_{ij}(t+\tau)]="
        r"R_{ii}(\tau)R_{jj}(\tau)+R_{ij}(\tau)R_{ji}(\tau)"
    )
    st.markdown(
        r"For a symmetric bivariate AR(1), $\rho_x(k)=\phi^k$ while the centered edge has "
        r"$\rho_e(k)=\phi^{2k}$."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    phi_3 = c1.slider("AR coefficient φ", 0.00, 0.98, 0.85, 0.01, key="phi3")
    rho_3 = c2.slider("Cross-correlation ρ", -0.80, 0.80, 0.35, 0.05, key="rho3")
    tr_3 = c3.slider("TR (s)", 0.40, 2.00, 0.72, 0.04, key="tr3")
    n_time_3 = c4.slider("Number of frames", 1000, 20000, 6000, 500, key="n3")
    pct_3 = c5.slider("Event percentile", 80, 99, 95, 1, key="pct3")

    x3, y3 = cached_ar1_pair(n_time_3, phi_3, rho_3, seed=int(global_seed) + 33)
    edge3 = x3 * y3
    centered3 = edge3 - np.mean(edge3)
    max_lag_frames3 = min(250, n_time_3 // 10)
    lags3 = np.arange(max_lag_frames3 + 1)
    acf_x3 = autocorrelation(x3, max_lag_frames3)
    acf_y3 = autocorrelation(y3, max_lag_frames3)
    acf_e3 = autocorrelation(centered3, max_lag_frames3)
    theory_node3 = phi_3**lags3
    theory_edge3 = phi_3 ** (2 * lags3)

    threshold3 = float(np.quantile(np.abs(centered3), pct_3 / 100.0))
    event_mask3 = np.abs(centered3) >= threshold3
    runs3 = contiguous_run_lengths(event_mask3)
    view3 = event_window(centered3, tr_3, 120.0)
    t3 = np.arange(n_time_3) * tr_3

    if phi_3 > 0:
        node_half3 = math.log(0.5) / math.log(phi_3) * tr_3
        edge_half3 = math.log(0.5) / math.log(phi_3**2) * tr_3
    else:
        node_half3 = edge_half3 = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Theoretical node half-life", f"{node_half3:.2f} s")
    m2.metric("Theoretical edge half-life", f"{edge_half3:.2f} s")
    m3.metric("Mean threshold-run length", f"{(runs3.mean() * tr_3 if runs3.size else 0):.2f} s")
    m4.metric("Number of event episodes", f"{runs3.size}")

    left, right = st.columns(2)
    with left:
        fig_ts3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
        fig_ts3.add_trace(go.Scatter(x=t3[view3], y=x3[view3], name="x"), row=1, col=1)
        fig_ts3.add_trace(go.Scatter(x=t3[view3], y=y3[view3], name="y"), row=1, col=1)
        fig_ts3.add_trace(go.Scatter(x=t3[view3], y=centered3[view3], name="centered edge"), row=2, col=1)
        local_events3 = np.flatnonzero(event_mask3[view3])
        if local_events3.size:
            fig_ts3.add_trace(
                go.Scatter(
                    x=t3[view3][local_events3],
                    y=centered3[view3][local_events3],
                    name="event frames",
                    mode="markers",
                ),
                row=2,
                col=1,
            )
        fig_ts3.update_xaxes(title_text="time (s)", row=2, col=1)
        st.plotly_chart(
            finish_figure(fig_ts3, height=560, title="Smooth nodes, faster product fluctuations"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        fig_acf3 = go.Figure()
        fig_acf3.add_trace(go.Scatter(x=lags3 * tr_3, y=acf_x3, name="sample ACF x"))
        fig_acf3.add_trace(go.Scatter(x=lags3 * tr_3, y=acf_y3, name="sample ACF y"))
        fig_acf3.add_trace(go.Scatter(x=lags3 * tr_3, y=acf_e3, name="sample ACF edge"))
        fig_acf3.add_trace(go.Scatter(x=lags3 * tr_3, y=theory_node3, name="theory φᵏ", line=dict(dash="dot")))
        fig_acf3.add_trace(
            go.Scatter(x=lags3 * tr_3, y=theory_edge3, name="theory φ²ᵏ", line=dict(dash="dash"))
        )
        fig_acf3.update_xaxes(title="lag (s)")
        fig_acf3.update_yaxes(title="autocorrelation", range=[-0.2, 1.02])
        st.plotly_chart(
            finish_figure(fig_acf3, height=560, title="The edge inherits products of lag-covariances"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )

    if runs3.size:
        fig_runs3 = go.Figure(go.Histogram(x=runs3 * tr_3, nbinsx=min(30, max(5, runs3.size // 3))))
        fig_runs3.update_xaxes(title="duration of contiguous threshold episode (s)")
        fig_runs3.update_yaxes(title="count")
        st.plotly_chart(
            finish_figure(fig_runs3, height=330, title="Autocorrelation turns threshold crossings into multi-frame episodes"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )


with tabs[3]:
    st.subheader("A stationary modular covariance can generate high-RSS cofluctuation events")
    st.markdown(
        "This panel has **no switching covariance**. Global, module, and private factors have fixed loadings. "
        "Nevertheless, extreme outer products occur, align with dominant covariance eigendirections, and can "
        "recover recognizable network topology."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    modules_4 = c1.slider("Modules", 2, 6, 4, 1, key="modules4")
    per_module_4 = c2.slider("Nodes/module", 4, 12, 8, 1, key="per4")
    within_4 = c3.slider("Within-module r", 0.20, 0.80, 0.45, 0.05, key="within4")
    between_4 = c4.slider("Between-module r", 0.00, 0.19, 0.08, 0.01, key="between4")
    phi_4 = c5.slider("Temporal AR φ", 0.00, 0.95, 0.60, 0.05, key="phi4")
    top_pct_4 = c6.slider("Selected top/bottom %", 1, 20, 5, 1, key="top4")
    n_time_4 = st.slider("Frames", 500, 6000, 2500, 250, key="ntime4")

    sim4, recon4 = cached_static_event_simulation(
        n_time=n_time_4,
        n_modules=modules_4,
        nodes_per_module=per_module_4,
        within_corr=within_4,
        between_corr=min(between_4, within_4),
        phi=phi_4,
        top_fraction=top_pct_4 / 100.0,
        seed=int(global_seed) + 44,
    )
    q4 = np.sum(sim4.data**2, axis=1)
    q4_sq = q4**2
    fourth4 = np.sum(sim4.data**4, axis=1)
    participation4 = np.divide(q4_sq - fourth4, q4_sq, out=np.zeros_like(q4), where=q4_sq > 1e-12)
    rss_formula4 = q4 * np.sqrt(np.maximum(participation4, 0.0) / 2.0)
    identity_error4 = float(np.max(np.abs(recon4.rss - rss_formula4)))
    corr_rss_q4 = safe_corr(recon4.rss, q4)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Full scan vs true topology", f"{recon4.full_score:.3f}")
    m2.metric("Top-RSS subset vs truth", f"{recon4.top_score:.3f}")
    m3.metric("Median random subset", f"{np.median(recon4.random_score_distribution):.3f}")
    m4.metric("Bottom-RSS subset vs truth", f"{recon4.bottom_score:.3f}")
    m5.metric("corr(RSS, ‖z‖²)", f"{corr_rss_q4:.3f}")

    h1, h2, h3 = st.columns(3)
    with h1:
        st.plotly_chart(
            matrix_heatmap(sim4.population_correlation, "Population correlation (fixed in time)"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with h2:
        st.plotly_chart(
            matrix_heatmap(recon4.top_pattern, "Mean outer product: top-RSS frames"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with h3:
        st.plotly_chart(
            matrix_heatmap(recon4.bottom_pattern, "Mean outer product: bottom-RSS frames"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )

    left, right = st.columns(2)
    with left:
        show_n4 = min(800, n_time_4)
        idx4 = np.arange(show_n4)
        top_mask4 = np.isin(idx4, recon4.top_indices)
        fig_rss4 = go.Figure()
        fig_rss4.add_trace(go.Scatter(x=idx4, y=recon4.rss[:show_n4], name="edge RSS", mode="lines"))
        if np.any(top_mask4):
            fig_rss4.add_trace(
                go.Scatter(
                    x=idx4[top_mask4],
                    y=recon4.rss[:show_n4][top_mask4],
                    name="top frames",
                    mode="markers",
                )
            )
        fig_rss4.update_xaxes(title="frame")
        fig_rss4.update_yaxes(title="whole-brain edge RSS")
        st.plotly_chart(
            finish_figure(fig_rss4, title="Events appear despite fixed covariance"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        fig_score4 = go.Figure(
            go.Histogram(x=recon4.random_score_distribution, nbinsx=35, name="random matched subsets")
        )
        fig_score4.add_vline(x=recon4.top_score, line_dash="dash", annotation_text="top-RSS")
        fig_score4.add_vline(x=recon4.bottom_score, line_dash="dot", annotation_text="bottom-RSS")
        fig_score4.add_vline(x=recon4.full_score, line_dash="solid", annotation_text="full scan")
        fig_score4.update_xaxes(title="edgewise correlation with population matrix")
        fig_score4.update_yaxes(title="count")
        st.plotly_chart(
            finish_figure(fig_score4, title="Matched random subsets provide the relevant baseline"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )

    fig_radial4 = go.Figure()
    fig_radial4.add_trace(
        go.Scattergl(
            x=q4,
            y=recon4.rss,
            mode="markers",
            name="frames",
            marker=dict(size=5, opacity=0.45),
            customdata=participation4,
            hovertemplate="‖z‖²=%{x:.2f}<br>RSS=%{y:.2f}<br>participation=%{customdata:.3f}<extra></extra>",
        )
    )
    fig_radial4.update_xaxes(title="radial energy ‖z(t)‖²")
    fig_radial4.update_yaxes(title="edge RSS")
    st.plotly_chart(
        finish_figure(fig_radial4, height=390, title="RSS combines radial energy with spatial participation"),
        use_container_width=True,
        config=PLOT_CONFIG,
    )
    st.caption(
        f"Numerical check of RSS² = ½[(Σzᵢ²)² − Σzᵢ⁴]: maximum absolute error = {identity_error4:.2e}."
    )


with tabs[4]:
    st.subheader("A seed-threshold CAP recovers a correlation map under a static Gaussian model")
    st.latex(
        r"\mathbb E[\mathbf z\mid z_s>u]="
        r"R_{\cdot s}\,\frac{\varphi(u)}{1-\Phi(u)}"
    )
    st.markdown(
        "The conditional mean is proportional to the seed column of the correlation matrix. This does not "
        "make CAP clustering uninformative; it shows that basic map recovery from high-seed frames is already "
        "expected without discrete state switching."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    modules_5 = c1.slider("Modules", 2, 6, 4, 1, key="modules5")
    per_module_5 = c2.slider("Nodes/module", 4, 12, 8, 1, key="per5")
    threshold_5 = c3.slider("Seed threshold u (z)", 0.0, 2.6, 1.0, 0.1, key="threshold5")
    phi_5 = c4.slider("Temporal AR φ", 0.0, 0.95, 0.40, 0.05, key="phi5")
    n_time_5 = c5.slider("Frames", 3000, 20000, 8000, 500, key="ntime5")

    n_nodes_5 = modules_5 * per_module_5
    seed_node_5 = st.slider("Seed node", 0, n_nodes_5 - 1, 0, 1, key="seed5")
    sim5, cap5 = cached_cap_simulation(
        n_time=n_time_5,
        n_modules=modules_5,
        nodes_per_module=per_module_5,
        phi=phi_5,
        seed_node=seed_node_5,
        threshold_z=threshold_5,
        seed=int(global_seed) + 55,
    )
    selected_pct5 = 100.0 * np.mean(cap5.selected)
    expected_pct5 = 100.0 * stats.norm.sf(threshold_5)
    cap_corr5 = safe_corr(cap5.cap, cap5.theoretical_cap)
    normalized_corr5 = safe_corr(cap5.cap / cap5.inverse_mills_ratio, cap5.population_fc_seed)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected frames", f"{selected_pct5:.2f}%")
    m2.metric("Gaussian expectation", f"{expected_pct5:.2f}%")
    m3.metric("CAP vs theoretical mean", f"{cap_corr5:.3f}")
    m4.metric("CAP/λ vs seed FC", f"{normalized_corr5:.3f}")

    node_index5 = np.arange(n_nodes_5)
    fig_cap5 = go.Figure()
    fig_cap5.add_trace(go.Bar(x=node_index5, y=cap5.cap, name="observed CAP"))
    fig_cap5.add_trace(
        go.Scatter(x=node_index5, y=cap5.theoretical_cap, name="Gaussian conditional mean", mode="lines+markers")
    )
    for boundary in range(per_module_5, n_nodes_5, per_module_5):
        fig_cap5.add_vline(x=boundary - 0.5, line_dash="dot")
    fig_cap5.update_xaxes(title="node index")
    fig_cap5.update_yaxes(title="conditional mean z")
    st.plotly_chart(
        finish_figure(fig_cap5, title="Observed CAP and static-Gaussian prediction"),
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    left, right = st.columns(2)
    with left:
        lo5 = min(np.min(cap5.theoretical_cap), np.min(cap5.cap))
        hi5 = max(np.max(cap5.theoretical_cap), np.max(cap5.cap))
        pad5 = 0.05 * max(hi5 - lo5, 1e-6)
        fig_scatter5 = go.Figure()
        fig_scatter5.add_trace(
            go.Scatter(
                x=cap5.theoretical_cap,
                y=cap5.cap,
                mode="markers+text",
                text=[str(i) if i == seed_node_5 else "" for i in node_index5],
                textposition="top center",
                name="nodes",
            )
        )
        fig_scatter5.add_trace(
            go.Scatter(x=[lo5 - pad5, hi5 + pad5], y=[lo5 - pad5, hi5 + pad5], name="identity", mode="lines")
        )
        fig_scatter5.update_xaxes(title="theoretical conditional mean")
        fig_scatter5.update_yaxes(title="observed CAP")
        st.plotly_chart(
            finish_figure(fig_scatter5, title="Conditional-Gaussian prediction node by node"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        seed_ts5 = sim5.data[:, seed_node_5]
        show5 = min(800, n_time_5)
        fig_seed5 = go.Figure()
        fig_seed5.add_trace(go.Scatter(x=np.arange(show5), y=seed_ts5[:show5], name="seed z", mode="lines"))
        mask5 = cap5.selected[:show5]
        if np.any(mask5):
            fig_seed5.add_trace(
                go.Scatter(
                    x=np.flatnonzero(mask5),
                    y=seed_ts5[:show5][mask5],
                    name="selected frames",
                    mode="markers",
                )
            )
        fig_seed5.add_hline(y=threshold_5, line_dash="dot", annotation_text="threshold")
        fig_seed5.update_xaxes(title="frame")
        fig_seed5.update_yaxes(title="seed z")
        st.plotly_chart(
            finish_figure(fig_seed5, title="Thresholding converts a continuous process into a point set"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )


with tabs[5]:
    st.subheader("Sparse latent events can become smooth, low-frequency correlated BOLD")
    st.markdown(
        r"Here the latent input is explicitly punctate. Convolution with a double-gamma HRF produces smooth "
        r"signals, and shared impulses produce low-frequency cross-spectral covariance."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    duration_6 = c1.slider("Duration (s)", 300, 2400, 1200, 60, key="duration6")
    tr_6 = c2.slider("TR (s)", 0.25, 2.00, 0.50, 0.05, key="tr6")
    rate_6 = c3.slider("Events/minute", 1.0, 24.0, 8.0, 0.5, key="rate6")
    shared_6 = c4.slider("Shared-event fraction", 0.0, 1.0, 0.55, 0.05, key="shared6")
    hrf_width_6 = c5.slider("HRF width scale", 0.5, 2.0, 1.0, 0.1, key="hrf6")
    noise_6 = c6.slider("Observation noise", 0.0, 1.5, 0.30, 0.05, key="noise6")

    sim6 = cached_event_bold(
        duration_s=duration_6,
        tr=tr_6,
        events_per_minute=rate_6,
        shared_fraction=shared_6,
        noise_sd=noise_6,
        hrf_width=hrf_width_6,
        seed=int(global_seed) + 66,
    )
    edge6 = sim6.bold_1 * sim6.bold_2
    f_b1_6, p_b1_6 = welch_psd(sim6.bold_1, tr_6)
    f_b2_6, p_b2_6 = welch_psd(sim6.bold_2, tr_6)
    f_e6, p_e6 = welch_psd(edge6 - np.mean(edge6), tr_6)
    f_c6, cospec6, cumulative6 = cospectrum_and_cumulative(sim6.bold_1, sim6.bold_2, tr_6)
    frac01_6 = fraction_at_frequency(f_c6, cumulative6, 0.10)
    frac02_6 = fraction_at_frequency(f_c6, cumulative6, 0.20)
    shared_count6 = int(np.count_nonzero(sim6.shared_events))
    sample_r6 = float(np.corrcoef(sim6.bold_1, sim6.bold_2)[0, 1])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BOLD correlation", f"{sample_r6:.3f}")
    m2.metric("Shared impulses", f"{shared_count6}")
    m3.metric("Signed covariance ≤ 0.10 Hz", format_spectral_fraction(frac01_6, sample_r6))
    m4.metric("Signed covariance ≤ 0.20 Hz", format_spectral_fraction(frac02_6, sample_r6))

    show_s6 = min(220.0, duration_6)
    view6 = sim6.time <= show_s6
    fig_event6 = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Sparse latent inputs", "HRF-convolved BOLD", "BOLD edge product"),
    )
    fig_event6.add_trace(go.Bar(x=sim6.time[view6], y=sim6.neural_1[view6], name="events 1"), row=1, col=1)
    fig_event6.add_trace(go.Bar(x=sim6.time[view6], y=-sim6.neural_2[view6], name="events 2 (reflected)"), row=1, col=1)
    fig_event6.add_trace(go.Scatter(x=sim6.time[view6], y=sim6.bold_1[view6], name="BOLD 1"), row=2, col=1)
    fig_event6.add_trace(go.Scatter(x=sim6.time[view6], y=sim6.bold_2[view6], name="BOLD 2"), row=2, col=1)
    fig_event6.add_trace(go.Scatter(x=sim6.time[view6], y=edge6[view6], name="edge"), row=3, col=1)
    fig_event6.update_xaxes(title_text="time (s)", row=3, col=1)
    st.plotly_chart(finish_figure(fig_event6, height=720), use_container_width=True, config=PLOT_CONFIG)

    left, right = st.columns(2)
    with left:
        fig_psd6 = go.Figure()
        fig_psd6.add_trace(go.Scatter(x=f_b1_6, y=p_b1_6 / integrate.trapezoid(p_b1_6, f_b1_6), name="BOLD 1"))
        fig_psd6.add_trace(go.Scatter(x=f_b2_6, y=p_b2_6 / integrate.trapezoid(p_b2_6, f_b2_6), name="BOLD 2"))
        fig_psd6.add_trace(go.Scatter(x=f_e6, y=p_e6 / integrate.trapezoid(p_e6, f_e6), name="edge"))
        fig_psd6.update_xaxes(title="frequency (Hz)", range=[0, min(0.4, 0.5 / tr_6)])
        fig_psd6.update_yaxes(title="area-normalized PSD")
        st.plotly_chart(
            finish_figure(fig_psd6, title="Hemodynamic convolution concentrates nodal power at low frequency"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with right:
        fig_cov6 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
        fig_cov6.add_trace(go.Scatter(x=f_c6, y=cospec6, name="co-spectrum"), row=1, col=1)
        fig_cov6.add_trace(go.Scatter(x=f_c6, y=cumulative6, name="cumulative fraction"), row=2, col=1)
        fig_cov6.add_hline(y=1.0, line_dash="dot", row=2, col=1)
        fig_cov6.add_vline(x=0.10, line_dash="dot", row=2, col=1)
        fig_cov6.add_vline(x=0.20, line_dash="dash", row=2, col=1)
        fig_cov6.update_xaxes(title_text="frequency (Hz)", range=[0, min(0.4, 0.5 / tr_6)], row=2, col=1)
        fig_cov6.update_yaxes(title_text="co-spectrum", row=1, col=1)
        fig_cov6.update_yaxes(title_text="fraction of signed covariance", row=2, col=1)
        st.plotly_chart(
            finish_figure(fig_cov6, title="Sparse causes can yield slow covariance"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )

    hrf_col, export_col = st.columns([2, 1])
    with hrf_col:
        fig_hrf6 = go.Figure(go.Scatter(x=sim6.hrf_time, y=sim6.hrf, name="HRF"))
        fig_hrf6.update_xaxes(title="time after impulse (s)")
        fig_hrf6.update_yaxes(title="normalized response")
        st.plotly_chart(
            finish_figure(fig_hrf6, height=330, title="Hemodynamic kernel"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with export_col:
        st.markdown(
            "Try narrowing or widening the HRF, eliminating shared events, and increasing observation noise. "
            "The latent process remains punctate, while the observed spectral and correlational descriptions change."
        )
        download_csv(
            "Download latent events and BOLD",
            pd.DataFrame(
                {
                    "time_s": sim6.time,
                    "neural_1": sim6.neural_1,
                    "neural_2": sim6.neural_2,
                    "bold_1": sim6.bold_1,
                    "bold_2": sim6.bold_2,
                    "edge": edge6,
                }
            ),
            "sparse_events_to_bold.csv",
            "download_tab6",
        )


with tabs[6]:
    st.subheader("Suggested experiments and interpretation")
    st.markdown(
        r"""
### A sequence that isolates the main ideas

1. **Tab 1:** hold the node cutoff at 0.08 Hz and increase the event threshold. The point set becomes sparser even though the underlying process has not changed.
2. **Tab 2:** set both carriers to 0.05 Hz. The real edge contains DC and 0.10 Hz, not 0.05 Hz. Then separate the carriers slightly and watch the beat and sum terms.
3. **Tab 3:** increase φ. Node and edge processes both become smoother, but the edge ACF still follows approximately the square of the nodal ACF.
4. **Tab 4:** set φ to zero. High-RSS events remain, showing that temporal autocorrelation is not required for heavy-tailed framewise leverage. Then increase φ to see those frames cluster into episodes.
5. **Tab 5:** raise the seed threshold. The CAP becomes based on fewer frames but its expected spatial direction remains the seed correlation map.
6. **Tab 6:** begin with shared events and a broad HRF, then set the shared fraction to zero. This separates low-frequency nodal power from low-frequency cross-covariance.

### What each demonstration does—and does not—show

- A static Gaussian null can produce bursty products, but this does **not** prove empirical events are Gaussian or biologically uninteresting.
- Event-frame dominance establishes unequal leverage on covariance estimation; it does **not** by itself establish a switching latent coupling parameter.
- Sparse latent inputs can generate slow BOLD, but BOLD alone may not identify whether sparsity originated neurally, physiologically, or through thresholding a continuous process.
- The most diagnostic empirical targets are residual fourth-order structure, event timing and sequence beyond a full cross-spectral null, external locking, propagation, and multimodal validation.

### Core identities implemented here

\[
\widehat R=\frac{1}{T}\sum_t \mathbf z_t\mathbf z_t^\top,
\qquad
\widehat r_{ij}=\frac{1}{T}\sum_t z_i(t)z_j(t)
\]

\[
\operatorname{Cov}[z_i(t)z_j(t),z_i(t+\tau)z_j(t+\tau)]
=R_{ii}(\tau)R_{jj}(\tau)+R_{ij}(\tau)R_{ji}(\tau)
\]

\[
\mathrm{RSS}(t)^2
=\frac12\left[\left(\sum_i z_i(t)^2\right)^2-\sum_i z_i(t)^4\right]
\]

\[
\mathbb E[\mathbf z\mid z_s>u]
=R_{\cdot s}\frac{\varphi(u)}{1-\Phi(u)}.
\]
"""
    )
    st.warning(
        "These are explanatory simulations, not validated preprocessing or inferential software. "
        "For empirical work, preserve the full preprocessing history, account for censoring and physiology, "
        "use held-out data where possible, and compare against nulls that retain the lower-order structure "
        "relevant to the event statistic."
    )
