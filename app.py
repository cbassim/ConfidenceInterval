import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import streamlit as st


APP_VERSION = "stable cumulative timing v11"


st.set_page_config(
    page_title="Confidence Interval Coverage Simulator",
    page_icon="📏",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.5rem;
        max-width: 1500px;
    }
    h1 {
        font-size: 1.5rem !important;
        margin-bottom: 0.1rem !important;
    }
    div[data-testid="stMetric"] {
        padding: 0.05rem 0.25rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.15rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class Settings:
    true_mean: float
    true_sd: float
    sample_size: int
    confidence_level: float
    population_shape: str


def settings_key(settings: Settings) -> Tuple:
    """Store only plain values in session state.

    Important Streamlit detail:
    Classes are re-created on every rerun. If a dataclass object is stored in
    st.session_state, equality can fail on the next rerun because the old object
    belongs to the previous class definition. This made older versions reset
    history every time a button was clicked. A plain tuple avoids that.
    """
    return (
        round(float(settings.true_mean), 10),
        round(float(settings.true_sd), 10),
        int(settings.sample_size),
        round(float(settings.confidence_level), 10),
        str(settings.population_shape),
    )


def initialize_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "running_loop" not in st.session_state:
        st.session_state.running_loop = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_added" not in st.session_state:
        st.session_state.last_added = 0
    if "active_settings_key" not in st.session_state:
        st.session_state.active_settings_key = None
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng()


def reset_history() -> None:
    st.session_state.history = []
    st.session_state.running_loop = False
    st.session_state.last_result = None
    st.session_state.last_added = 0
    st.session_state.rng = np.random.default_rng()


def draw_sample(settings: Settings) -> np.ndarray:
    rng = st.session_state.rng
    mu = settings.true_mean
    sigma = settings.true_sd
    n = settings.sample_size

    if settings.population_shape == "Normal":
        return rng.normal(loc=mu, scale=sigma, size=n)

    if settings.population_shape == "Right-skewed":
        return rng.exponential(scale=sigma, size=n) + (mu - sigma)

    if settings.population_shape == "Uniform":
        half_width = np.sqrt(3) * sigma
        return rng.uniform(low=mu - half_width, high=mu + half_width, size=n)

    raise ValueError(f"Unknown population shape: {settings.population_shape}")


def simulate_one_interval(settings: Settings) -> Dict:
    sample = draw_sample(settings)
    n = settings.sample_size

    sample_mean = float(np.mean(sample))
    sample_sd = float(np.std(sample, ddof=1))
    sem = sample_sd / np.sqrt(n)

    alpha = 1 - settings.confidence_level
    t_star = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    margin = float(t_star * sem)

    lower = sample_mean - margin
    upper = sample_mean + margin
    covered = bool(lower <= settings.true_mean <= upper)

    return {
        "sample": sample,
        "sample_mean": sample_mean,
        "sample_sd": sample_sd,
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
    for h in history[-250:]:
        values.extend([h["lower"], h["upper"], h["sample_mean"]])

    if current is not None:
        values.extend([current["lower"], current["upper"], current["sample_mean"]])
        values.extend(list(current["sample"]))

    left = min(values) - 0.15 * sigma
    right = max(values) + 0.15 * sigma
    return np.linspace(left, right, 500)


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


def tally(history: List[Dict]) -> Tuple[int, int, int, float]:
    total = len(history)
    covered = sum(1 for h in history if h["covered"])
    missed = total - covered
    coverage = covered / total if total else 0.0
    return total, covered, missed, coverage


SPEED_OPTIONS = ["Slow", "Medium", "Fast", "Super fast", "Turbo"]

SPEED_SPECS = {
    # Slow/Medium/Fast show the falling-and-growing animation.
    # Super fast adds visible chunks into the lower plot, so it is much faster but not instant.
    # Turbo adds the entire requested batch at once and redraws once.
    "Slow": {"mode": "animated", "frames": 30, "frame_delay": 0.085, "after_interval_delay": 0.14, "loop_batch": 1},
    "Medium": {"mode": "animated", "frames": 18, "frame_delay": 0.055, "after_interval_delay": 0.09, "loop_batch": 1},
    "Fast": {"mode": "animated", "frames": 5, "frame_delay": 0.015, "after_interval_delay": 0.020, "loop_batch": 1},
    "Super fast": {"mode": "quick_batch", "frames": 1, "frame_delay": 0.000, "after_interval_delay": 0.040, "loop_batch": 10, "chunk_size": 5},
    "Turbo": {"mode": "batch", "frames": 1, "frame_delay": 0.000, "after_interval_delay": 0.00, "loop_batch": 50},
}


def render_tally(history: List[Dict], settings: Settings, just_added: int = 0) -> None:
    total, covered, missed, coverage = tally(history)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total intervals", f"{total}")
    c2.metric("Covered true μ", f"{covered}")
    c3.metric("Missed true μ", f"{missed}")
    c4.metric("Running coverage", "—" if total == 0 else f"{100 * coverage:.1f}%")
    c5.metric("Target level", f"{100 * settings.confidence_level:.0f}%")
    c6.metric("Just added", f"{just_added}")

    if total:
        st.caption(
            f"Latest interval #{total}: "
            f"[{history[-1]['lower']:.3f}, {history[-1]['upper']:.3f}] "
            f"— {'covered' if history[-1]['covered'] else 'missed'} the true mean."
        )
    else:
        st.caption("Click Add 1, Add 10, Add 100, or Start loop. Counts accumulate until Reset or settings change.")


def render_figure(
    settings: Settings,
    history: List[Dict],
    current: Optional[Dict] = None,
    progress: float = 1.0,
    max_rows: int = 100,
) -> plt.Figure:
    progress = float(np.clip(progress, 0, 1))

    x = population_grid(settings, current, history)
    y = population_pdf(x, settings)
    max_pdf = max(float(np.max(y)), 1e-9)
    mu = settings.true_mean

    total_after_current = len(history) + (1 if current is not None else 0)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.82, 1.65], "hspace": 0.30},
    )

    ax_pop, ax_sample, ax_ci = axes

    ax_pop.plot(x, y, color="#1f77b4", linewidth=2)
    ax_pop.axvline(mu, color="#d62728", linestyle="--", linewidth=2, label="true mean μ")
    ax_pop.set_ylabel("density")
    ax_pop.set_title("1. Draw a random sample", fontsize=12)
    ax_pop.set_ylim(-0.04 * max_pdf, 1.18 * max_pdf)
    ax_pop.legend(loc="upper right", frameon=False, fontsize=9)

    if current is not None:
        sample = np.asarray(current["sample"])
        y_start = population_pdf(sample, settings)
        y_floor = np.full_like(sample, -0.018 * max_pdf, dtype=float)
        y_points = (1 - progress) * y_start + progress * y_floor
        ax_pop.scatter(sample, y_points, color="#333333", s=34, alpha=0.9, zorder=5)

    ax_sample.axvline(mu, color="#d62728", linestyle="--", linewidth=2)
    ax_sample.set_title("2. Compute x̄ and grow the t interval", fontsize=12)
    ax_sample.set_yticks([])
    ax_sample.set_ylim(-0.15, 1.08)
    ax_sample.text(mu, 1.0, "true μ", color="#d62728", ha="center", va="bottom", fontsize=9)

    if current is not None:
        sample = np.asarray(current["sample"])
        display_rng = np.random.default_rng(12345)
        jitter = display_rng.uniform(-0.04, 0.04, size=len(sample))
        sample_y = 0.70 + jitter

        alpha_points = 0.25 + 0.75 * progress
        ax_sample.scatter(sample, sample_y, color="#333333", s=26, alpha=alpha_points, zorder=4)

        ci_progress = np.clip((progress - 0.30) / 0.70, 0, 1)
        center = current["sample_mean"]
        lower = center - current["margin"] * ci_progress
        upper = center + current["margin"] * ci_progress
        ci_color = "#2ca02c" if current["covered"] else "#d62728"

        ax_sample.hlines(0.32, lower, upper, color=ci_color, linewidth=4.5)
        ax_sample.scatter([center], [0.32], color="black", s=42, zorder=5)
        ax_sample.text(center, 0.15, "x̄", ha="center", va="top", fontsize=11)

    ax_ci.axvline(mu, color="#d62728", linestyle="--", linewidth=2)
    ax_ci.set_title(f"3. Stack intervals and count coverage — total simulated: {total_after_current}", fontsize=12)
    ax_ci.set_ylabel("cumulative interval #")
    ax_ci.set_xlabel("measurement value")

    history_start = max(0, len(history) - max_rows)
    shown_history = history[history_start:]

    for local_i, h in enumerate(shown_history):
        absolute_i = history_start + local_i + 1
        color = "#2ca02c" if h["covered"] else "#d62728"
        ax_ci.hlines(absolute_i, h["lower"], h["upper"], color=color, linewidth=2.1, alpha=0.86)
        ax_ci.scatter([h["sample_mean"]], [absolute_i], color="black", s=12, alpha=0.75)

    if current is not None:
        current_y = len(history) + 1
        ci_progress = np.clip((progress - 0.30) / 0.70, 0, 1)
        center = current["sample_mean"]
        lower = center - current["margin"] * ci_progress
        upper = center + current["margin"] * ci_progress
        color = "#2ca02c" if current["covered"] else "#d62728"
        ax_ci.hlines(current_y, lower, upper, color=color, linewidth=4.0, alpha=0.95)
        ax_ci.scatter([center], [current_y], color="black", s=26, zorder=5)

    if total_after_current == 0:
        y_bottom, y_top = 0, 6
    elif total_after_current <= max_rows:
        y_bottom, y_top = 0, max(6, total_after_current + 1)
    else:
        y_bottom = max(1, total_after_current - max_rows + 1) - 1
        y_top = total_after_current + 1

    ax_ci.set_ylim(y_bottom, y_top)
    ax_ci.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    if total_after_current > 0:
        first_visible = max(1, total_after_current - max_rows + 1)
        ax_ci.text(x[0], first_visible, f"{first_visible}", va="center", fontsize=8, alpha=0.65)
        ax_ci.text(x[0], total_after_current, f"{total_after_current}", va="center", fontsize=8, alpha=0.65)

    for ax in axes:
        ax.grid(alpha=0.16)

    fig.tight_layout()
    return fig


def redraw_static(settings: Settings, plot_placeholder, tally_placeholder, max_rows: int, just_added: int = 0) -> None:
    with plot_placeholder.container():
        fig = render_figure(settings=settings, history=st.session_state.history, current=None, progress=1.0, max_rows=max_rows)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with tally_placeholder.container():
        render_tally(st.session_state.history, settings, just_added=just_added)


def animate_then_append(settings: Settings, result: Dict, plot_placeholder, tally_placeholder, max_rows: int, speed_spec: Dict) -> None:
    mode = speed_spec.get("mode", "animated")
    frames = int(speed_spec["frames"])
    frame_delay = float(speed_spec["frame_delay"])

    if mode == "quick_ci":
        # A short visible build: the sample has mostly fallen already, and the CI
        # grows quickly into the lower stack. This is much faster than Fast, but
        # still lets students see each new interval being made.
        progress_values = [0.45, 0.70, 0.88, 1.00]

        for progress in progress_values:
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

            with tally_placeholder.container():
                render_tally(st.session_state.history, settings, just_added=0)

            time.sleep(frame_delay)

        st.session_state.history.append(compact_result(result))
        redraw_static(settings, plot_placeholder, tally_placeholder, max_rows, just_added=1)
        time.sleep(float(speed_spec["after_interval_delay"]))
        return

    if frames <= 1:
        st.session_state.history.append(compact_result(result))
        return

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

        with tally_placeholder.container():
            render_tally(st.session_state.history, settings, just_added=0)

        time.sleep(frame_delay)

    st.session_state.history.append(compact_result(result))

    redraw_static(settings, plot_placeholder, tally_placeholder, max_rows, just_added=1)
    time.sleep(float(speed_spec["after_interval_delay"]))


def run_intervals(number_to_add: int, settings: Settings, plot_placeholder, tally_placeholder, max_rows: int, speed_name: str) -> None:
    speed_spec = SPEED_SPECS[speed_name]

    if speed_spec.get("mode") == "batch":
        new_results = [compact_result(simulate_one_interval(settings)) for _ in range(number_to_add)]
        st.session_state.history.extend(new_results)
        redraw_static(settings, plot_placeholder, tally_placeholder, max_rows, just_added=number_to_add)
        return

    if speed_spec.get("mode") == "quick_batch":
        # Add intervals in visible chunks. This is much faster than the full animation,
        # but still lets the lower plot and tally visibly accumulate.
        chunk_size = int(speed_spec.get("chunk_size", 5))
        added = 0

        while added < number_to_add:
            batch_size = min(chunk_size, number_to_add - added)

            new_results = [
                compact_result(simulate_one_interval(settings))
                for _ in range(batch_size)
            ]

            st.session_state.history.extend(new_results)
            added += batch_size

            redraw_static(
                settings,
                plot_placeholder,
                tally_placeholder,
                max_rows,
                just_added=added,
            )

            time.sleep(float(speed_spec["after_interval_delay"]))

        return

    for _ in range(number_to_add):
        result = simulate_one_interval(settings)
        st.session_state.last_result = result
        animate_then_append(
            settings=settings,
            result=result,
            plot_placeholder=plot_placeholder,
            tally_placeholder=tally_placeholder,
            max_rows=max_rows,
            speed_spec=speed_spec,
        )

    redraw_static(settings, plot_placeholder, tally_placeholder, max_rows, just_added=number_to_add)


initialize_state()

top_left, top_right = st.columns([2.8, 1])
with top_left:
    st.title("Animated Confidence Interval Coverage Simulator")
    st.caption(
        "Repeated samples fall from a known population into t-based confidence intervals. "
        "The running tally accumulates until Reset or a statistical setting changes."
    )
with top_right:
    st.markdown(f"**Version:** `{APP_VERSION}`")

with st.sidebar:
    st.header("Settings")

    population_shape = st.selectbox(
        "Population shape",
        options=["Normal", "Right-skewed", "Uniform"],
        index=0,
    )

    true_mean = st.number_input("True mean, μ", value=102.0, step=0.1, format="%.3f")
    true_sd = st.number_input("True standard deviation, σ", value=1.2, min_value=0.001, step=0.1, format="%.3f")
    sample_size = st.slider("Sample size, n", min_value=3, max_value=100, value=10, step=1)
    confidence_pct = st.slider("Confidence level", min_value=80, max_value=99, value=95, step=1)

    st.divider()

    speed_name = st.select_slider(
        "Speed",
        options=SPEED_OPTIONS,
        value="Fast",
        help="Fast shows the full animation. Super fast adds visible chunks quickly. Turbo adds the batch instantly.",
    )

    max_rows = st.slider(
        "Intervals shown",
        min_value=25,
        max_value=250,
        value=100,
        step=25,
        help="The lower plot shows the most recent intervals. The y-axis uses cumulative interval number.",
    )

    st.caption(
        f"Loop batch at this speed: {SPEED_SPECS[speed_name]['loop_batch']} interval(s) per refresh. "
        "Add buttons accumulate until Reset."
    )

settings = Settings(
    true_mean=float(true_mean),
    true_sd=float(true_sd),
    sample_size=int(sample_size),
    confidence_level=float(confidence_pct) / 100,
    population_shape=population_shape,
)

current_settings_key = settings_key(settings)

if st.session_state.active_settings_key is None:
    st.session_state.active_settings_key = current_settings_key
elif st.session_state.active_settings_key != current_settings_key:
    reset_history()
    st.session_state.active_settings_key = current_settings_key

b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1.35, 1])

add_one = b1.button("Add 1", use_container_width=True)
add_ten = b2.button("Add 10", use_container_width=True)
add_hundred = b3.button("Add 100", use_container_width=True)

if st.session_state.running_loop:
    stop_loop = b4.button("Stop loop", type="primary", use_container_width=True)
    start_loop = False
else:
    start_loop = b4.button("Start loop", type="primary", use_container_width=True)
    stop_loop = False

reset = b5.button("Reset", use_container_width=True)

if reset:
    reset_history()

if start_loop:
    st.session_state.running_loop = True

if stop_loop:
    st.session_state.running_loop = False

plot_placeholder = st.empty()
tally_placeholder = st.empty()

redraw_static(settings, plot_placeholder, tally_placeholder, max_rows, just_added=0)

if add_one:
    st.session_state.running_loop = False
    run_intervals(1, settings, plot_placeholder, tally_placeholder, max_rows, speed_name)

elif add_ten:
    st.session_state.running_loop = False
    run_intervals(10, settings, plot_placeholder, tally_placeholder, max_rows, speed_name)

elif add_hundred:
    st.session_state.running_loop = False
    run_intervals(100, settings, plot_placeholder, tally_placeholder, max_rows, speed_name)

elif st.session_state.running_loop:
    batch = int(SPEED_SPECS[speed_name]["loop_batch"])
    run_intervals(batch, settings, plot_placeholder, tally_placeholder, max_rows, speed_name)
    time.sleep(0.05)
    st.rerun()

with st.expander("Recent intervals and download", expanded=False):
    df_history = history_to_frame(st.session_state.history)

    if df_history.empty:
        st.info("No intervals yet.")
    else:
        st.dataframe(df_history.tail(30), hide_index=True, use_container_width=True)
        st.download_button(
            "Download all simulated intervals as CSV",
            data=df_history.to_csv(index=False).encode("utf-8"),
            file_name="confidence_interval_simulation_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

with st.expander("Teaching note", expanded=False):
    st.markdown(
        f"""
        The interval shown is the usual one-sample t interval:

        $$
        \\bar{{x}} \\pm t^*_{{1-\\alpha/2,\\,n-1}}\\frac{{s}}{{\\sqrt{{n}}}}
        $$

        A single interval either covers the true mean or misses it. The confidence level describes the
        long-run behaviour of the interval method across repeated samples. With many repeated samples,
        the running coverage should drift toward about **{confidence_pct}%** when the assumptions behave well.
        """
    )
