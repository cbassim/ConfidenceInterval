import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st


# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Confidence Interval Coverage Simulator",
    page_icon="📏",
    layout="wide",
)


# ------------------------------------------------------------
# Simulation helpers
# ------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    true_mean: float
    true_sd: float
    sample_size: int
    confidence_level: float
    population_shape: str


def make_rng() -> np.random.Generator:
    """Create or retrieve a session-level random number generator."""
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng()
    return st.session_state.rng


def reset_history() -> None:
    st.session_state.history = []
    st.session_state.running = False
    st.session_state.last_result = None


def initialize_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "running" not in st.session_state:
        st.session_state.running = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "active_settings" not in st.session_state:
        st.session_state.active_settings = None


def draw_sample(settings: Settings) -> np.ndarray:
    """Sample from a population with the chosen true mean and true standard deviation."""
    rng = make_rng()
    mu = settings.true_mean
    sigma = settings.true_sd
    n = settings.sample_size

    if settings.population_shape == "Normal":
        return rng.normal(loc=mu, scale=sigma, size=n)

    if settings.population_shape == "Right-skewed":
        # Exponential has mean=scale and sd=scale. Shift it so the true mean is mu.
        return rng.exponential(scale=sigma, size=n) + (mu - sigma)

    if settings.population_shape == "Uniform":
        # Uniform(a,b) has mean=(a+b)/2 and sd=(b-a)/sqrt(12).
        half_width = np.sqrt(3) * sigma
        return rng.uniform(low=mu - half_width, high=mu + half_width, size=n)

    raise ValueError(f"Unknown population shape: {settings.population_shape}")


def simulate_one_interval(settings: Settings) -> Dict:
    sample = draw_sample(settings)
    n = settings.sample_size
    xbar = float(np.mean(sample))
    s = float(np.std(sample, ddof=1))
    sem = s / np.sqrt(n)
    alpha = 1 - settings.confidence_level
    t_star = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    margin = t_star * sem
    lower = xbar - margin
    upper = xbar + margin
    covered = bool(lower <= settings.true_mean <= upper)

    return {
        "sample": sample,
        "sample_mean": xbar,
        "sample_sd": s,
        "sem": sem,
        "t_star": t_star,
        "margin": margin,
        "lower": float(lower),
        "upper": float(upper),
        "covered": covered,
        "confidence_level": settings.confidence_level,
        "sample_size": n,
        "population_shape": settings.population_shape,
    }


def compact_result(result: Dict) -> Dict:
    """Keep only the fields needed for the running history."""
    return {
        "sample_mean": result["sample_mean"],
        "sample_sd": result["sample_sd"],
        "sem": result["sem"],
        "t_star": result["t_star"],
        "margin": result["margin"],
        "lower": result["lower"],
        "upper": result["upper"],
        "covered": result["covered"],
        "confidence_level": result["confidence_level"],
        "sample_size": result["sample_size"],
        "population_shape": result["population_shape"],
    }


def population_grid(settings: Settings, current: Optional[Dict], history: List[Dict]) -> np.ndarray:
    mu = settings.true_mean
    sigma = settings.true_sd

    if settings.population_shape == "Right-skewed":
        left = mu - 1.25 * sigma
        right = mu + 7.0 * sigma
    elif settings.population_shape == "Uniform":
        half_width = np.sqrt(3) * sigma
        left = mu - 2.0 * half_width
        right = mu + 2.0 * half_width
    else:
        left = mu - 4.5 * sigma
        right = mu + 4.5 * sigma

    values = [left, right, mu]
    for h in history[-150:]:
        values.extend([h["lower"], h["upper"], h["sample_mean"]])
    if current is not None:
        values.extend([current["lower"], current["upper"], current["sample_mean"]])
        values.extend(list(current["sample"]))

    left = min(values) - 0.15 * sigma
    right = max(values) + 0.15 * sigma
    return np.linspace(left, right, 500)


def population_pdf(x: np.ndarray, settings: Settings) -> np.ndarray:
    mu = settings.true_mean
    sigma = settings.true_sd

    if settings.population_shape == "Normal":
        return stats.norm.pdf(x, loc=mu, scale=sigma)

    if settings.population_shape == "Right-skewed":
        return stats.expon.pdf(x, loc=mu - sigma, scale=sigma)

    if settings.population_shape == "Uniform":
        half_width = np.sqrt(3) * sigma
        return stats.uniform.pdf(x, loc=mu - half_width, scale=2 * half_width)

    raise ValueError(f"Unknown population shape: {settings.population_shape}")


def history_to_frame(history: List[Dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history).copy()
    df.insert(0, "interval_number", np.arange(1, len(df) + 1))
    df["covered"] = df["covered"].map({True: "yes", False: "no"})
    return df[
        [
            "interval_number",
            "sample_mean",
            "sample_sd",
            "sem",
            "lower",
            "upper",
            "covered",
            "confidence_level",
            "sample_size",
            "population_shape",
        ]
    ]


# ------------------------------------------------------------
# Plotting and animation
# ------------------------------------------------------------
def render_metrics(history: List[Dict], current: Optional[Dict] = None, include_current: bool = False) -> None:
    covered_count = sum(1 for h in history if h["covered"])
    total_count = len(history)

    if current is not None and include_current:
        total_count += 1
        covered_count += int(current["covered"])

    miss_count = total_count - covered_count
    coverage_rate = covered_count / total_count if total_count else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intervals created", f"{total_count}")
    c2.metric("Covered true μ", f"{covered_count}")
    c3.metric("Missed true μ", f"{miss_count}")
    c4.metric("Running coverage", "—" if total_count == 0 else f"{100 * coverage_rate:.1f}%")

    if current is not None:
        status = "covers" if current["covered"] else "misses"
        st.markdown(
            f"Current interval: **[{current['lower']:.3f}, {current['upper']:.3f}]** "
            f"with sample mean **{current['sample_mean']:.3f}** — it **{status}** the true mean."
        )


def render_figure(
    settings: Settings,
    history: List[Dict],
    current: Optional[Dict] = None,
    progress: float = 1.0,
    max_rows: int = 100,
) -> plt.Figure:
    """Create the animated teaching figure.

    progress = 0: sample points are still up on the population distribution.
    progress = 1: sample has fallen into the confidence interval stack.
    """
    progress = float(np.clip(progress, 0, 1))
    x = population_grid(settings, current, history)
    y = population_pdf(x, settings)
    max_pdf = max(float(np.max(y)), 1e-9)
    mu = settings.true_mean

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.7]},
    )
    ax_pop, ax_sample, ax_ci = axes

    # --- Panel 1: true population and falling sample points ---
    ax_pop.plot(x, y, color="#1f77b4", linewidth=2)
    ax_pop.axvline(mu, color="#d62728", linestyle="--", linewidth=2, label="true mean μ")
    ax_pop.set_ylabel("density")
    ax_pop.set_title("1. Draw a random sample from the population")
    ax_pop.set_ylim(-0.03 * max_pdf, 1.20 * max_pdf)
    ax_pop.legend(loc="upper right", frameon=False)

    if current is not None:
        sample = np.asarray(current["sample"])
        y_start = population_pdf(sample, settings)
        y_floor = np.full_like(sample, -0.015 * max_pdf, dtype=float)
        y_points = (1 - progress) * y_start + progress * y_floor
        ax_pop.scatter(sample, y_points, color="#333333", s=42, alpha=0.9, zorder=5)

    # --- Panel 2: sample strip and CI forming around xbar ---
    ax_sample.axvline(mu, color="#d62728", linestyle="--", linewidth=2)
    ax_sample.set_title("2. Compute the sample mean and grow the t confidence interval")
    ax_sample.set_yticks([])
    ax_sample.set_ylim(-0.2, 1.15)
    ax_sample.text(mu, 1.03, "true μ", color="#d62728", ha="center", va="bottom")

    if current is not None:
        sample = np.asarray(current["sample"])
        # sample points settle on the sample strip as progress increases
        rng = np.random.default_rng(12345)  # stable jitter for display only
        jitter = rng.uniform(-0.05, 0.05, size=len(sample))
        sample_y = 0.72 + jitter
        alpha_points = 0.25 + 0.75 * progress
        ax_sample.scatter(sample, sample_y, color="#333333", s=36, alpha=alpha_points, zorder=4)

        ci_progress = np.clip((progress - 0.35) / 0.65, 0, 1)
        center = current["sample_mean"]
        lower = center - current["margin"] * ci_progress
        upper = center + current["margin"] * ci_progress
        ci_color = "#2ca02c" if current["covered"] else "#d62728"
        ax_sample.hlines(0.35, lower, upper, color=ci_color, linewidth=5)
        ax_sample.scatter([center], [0.35], color="black", s=55, zorder=5)
        ax_sample.text(center, 0.18, r"$\bar{x}$", ha="center", va="top", fontsize=12)
        if ci_progress > 0:
            ax_sample.text(lower, 0.47, "L", ha="center", va="bottom", fontsize=10)
            ax_sample.text(upper, 0.47, "U", ha="center", va="bottom", fontsize=10)

    # --- Panel 3: running stack of confidence intervals ---
    shown_history = history[-max_rows:]
    first_index = max(0, len(history) - max_rows)
    ax_ci.axvline(mu, color="#d62728", linestyle="--", linewidth=2)
    ax_ci.set_title("3. Stack the intervals and count how often they cover the true mean")
    ax_ci.set_ylabel("interval #")

    for local_i, h in enumerate(shown_history, start=1):
        absolute_i = first_index + local_i
        color = "#2ca02c" if h["covered"] else "#d62728"
        ax_ci.hlines(local_i, h["lower"], h["upper"], color=color, linewidth=2.2, alpha=0.85)
        ax_ci.scatter([h["sample_mean"]], [local_i], color="black", s=14, alpha=0.75)
        if local_i == 1 or local_i == len(shown_history) or absolute_i % 25 == 0:
            ax_ci.text(x[0], local_i, str(absolute_i), va="center", fontsize=8, alpha=0.65)

    if current is not None:
        ci_progress = np.clip((progress - 0.35) / 0.65, 0, 1)
        current_y = len(shown_history) + 1
        center = current["sample_mean"]
        lower = center - current["margin"] * ci_progress
        upper = center + current["margin"] * ci_progress
        color = "#2ca02c" if current["covered"] else "#d62728"
        ax_ci.hlines(current_y, lower, upper, color=color, linewidth=4.0, alpha=0.95)
        ax_ci.scatter([center], [current_y], color="black", s=28, zorder=5)
        ax_ci.text(x[0], current_y, str(len(history) + 1), va="center", fontsize=9, fontweight="bold")

    y_top = len(shown_history) + (1 if current is not None else 0) + 1
    ax_ci.set_ylim(0, max(6, y_top))
    ax_ci.set_xlabel("measurement value")

    for ax in axes:
        ax.grid(alpha=0.18)

    fig.tight_layout()
    return fig


def animate_interval(
    settings: Settings,
    result: Dict,
    plot_placeholder,
    metric_placeholder,
    frames: int,
    delay: float,
    max_rows: int,
) -> None:
    for frame in range(frames):
        progress = frame / max(frames - 1, 1)
        with plot_placeholder.container():
            fig = render_figure(
                settings=settings,
                history=st.session_state.history,
                current=result,
                progress=progress,
                max_rows=max_rows,
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        with metric_placeholder.container():
            render_metrics(st.session_state.history, current=result, include_current=(progress >= 0.98))

        time.sleep(delay)


# ------------------------------------------------------------
# App layout
# ------------------------------------------------------------
initialize_state()

st.title("Animated Confidence Interval Coverage Simulator")
st.caption(
    "Watch repeated samples fall from a known population distribution into t-based confidence intervals, "
    "then track how often the intervals cover the true mean."
)

with st.expander("Teaching idea", expanded=True):
    st.markdown(
        """
        In real data analysis, the true mean is unknown. In this simulation we **choose** the true mean, so we can check whether each confidence interval covers it.

        The interval used here is the usual one-sample t interval:
        """
    )
    st.latex(r"\bar{x} \pm t^*_{1-\alpha/2,\,n-1}\frac{s}{\sqrt{n}}")
    st.markdown(
        "A single realized interval either covers the true mean or it does not. The confidence level describes the **long-run behaviour of the method** across repeated samples."
    )

with st.sidebar:
    st.header("Simulation settings")
    population_shape = st.selectbox(
        "Population shape",
        options=["Normal", "Right-skewed", "Uniform"],
        index=0,
        help="Normal matches the original boiling-point example. The other shapes are useful for discussing robustness and sample size.",
    )
    true_mean = st.number_input("True mean, μ", value=102.0, step=0.1, format="%.3f")
    true_sd = st.number_input("True standard deviation, σ", value=1.2, min_value=0.001, step=0.1, format="%.3f")
    sample_size = st.slider("Sample size, n", min_value=3, max_value=100, value=10, step=1)
    confidence_pct = st.slider("Confidence level", min_value=80, max_value=99, value=95, step=1)
    confidence_level = confidence_pct / 100

    st.divider()
    animation_speed = st.select_slider(
        "Animation speed",
        options=["Slow", "Medium", "Fast"],
        value="Medium",
    )
    max_rows = st.slider("Maximum intervals shown", min_value=25, max_value=250, value=100, step=25)

settings = Settings(
    true_mean=float(true_mean),
    true_sd=float(true_sd),
    sample_size=int(sample_size),
    confidence_level=float(confidence_level),
    population_shape=population_shape,
)

# Reset when the user changes the simulation settings. Otherwise the running
# coverage would mix different confidence levels, sample sizes, or populations.
if st.session_state.active_settings is None:
    st.session_state.active_settings = settings
elif st.session_state.active_settings != settings:
    reset_history()
    st.session_state.active_settings = settings

if animation_speed == "Fast":
    frames, delay = 4, 0.045
elif animation_speed == "Slow":
    frames, delay = 12, 0.12
else:
    frames, delay = 8, 0.075

# Main control buttons
b1, b2, b3, b4 = st.columns([1, 1, 1.2, 1])
run_once = b1.button("Run once", use_container_width=True)
run_ten = b2.button("Run 10 times", use_container_width=True)

if st.session_state.running:
    stop_continuous = b3.button("Stop continuous", type="primary", use_container_width=True)
    start_continuous = False
else:
    start_continuous = b3.button("Start continuous", type="primary", use_container_width=True)
    stop_continuous = False

reset = b4.button("Reset", use_container_width=True)

if reset:
    reset_history()

if start_continuous:
    st.session_state.running = True

if stop_continuous:
    st.session_state.running = False

plot_placeholder = st.empty()
metric_placeholder = st.empty()

# Decide how many intervals to animate on this app run.
runs_requested = 0
if run_once:
    runs_requested = 1
elif run_ten:
    runs_requested = 10
elif st.session_state.running:
    runs_requested = 1

if runs_requested == 0:
    with plot_placeholder.container():
        fig = render_figure(settings=settings, history=st.session_state.history, current=None, max_rows=max_rows)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with metric_placeholder.container():
        render_metrics(st.session_state.history)
else:
    for _ in range(runs_requested):
        result = simulate_one_interval(settings)
        st.session_state.last_result = result
        animate_interval(
            settings=settings,
            result=result,
            plot_placeholder=plot_placeholder,
            metric_placeholder=metric_placeholder,
            frames=frames,
            delay=delay,
            max_rows=max_rows,
        )
        st.session_state.history.append(compact_result(result))

    with plot_placeholder.container():
        fig = render_figure(settings=settings, history=st.session_state.history, current=None, max_rows=max_rows)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with metric_placeholder.container():
        render_metrics(st.session_state.history)

# Continuous mode: after finishing one interval, rerun the app. The controls are
# redrawn each cycle, so the Stop button can interrupt the next cycle.
if st.session_state.running:
    time.sleep(0.10)
    st.rerun()

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Recent intervals")
    df_history = history_to_frame(st.session_state.history)
    if df_history.empty:
        st.info("Create intervals to see the running table.")
    else:
        st.dataframe(df_history.tail(20), hide_index=True, use_container_width=True)
        st.download_button(
            "Download all simulated intervals as CSV",
            data=df_history.to_csv(index=False).encode("utf-8"),
            file_name="confidence_interval_simulation_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

with right:
    st.subheader("What to notice")
    st.markdown(
        f"""
        - Higher confidence levels make intervals wider.
        - Lower confidence levels make intervals narrower, but miss the true mean more often.
        - With many repeated samples, the running coverage should drift toward about **{confidence_pct}%** when the method's assumptions behave well.
        - Try the right-skewed population with small and large sample sizes to discuss the CLT and robustness.
        """
    )
