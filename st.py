import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima_process import arma_generate_sample
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(layout="wide", page_title="Time Series Stationarity Concepts ")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #43a047;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .subsection-header {
        font-size: 22px;
        font-weight: bold;
        color: #7b1fa2;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .code-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Time Series Stationarity & Unit Root Tests by Dr Merwan Roudane </div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
sections = [
    "Introduction to Stationarity",
    "Types of Stationarity",
    "Visualizing Stationarity",
    "Unit Root Tests",
    "Cases in Unit Root Tests",
    "Structural Breaks",
    "Power of Unit Root Tests",
    "Interactive Examples"
]

selected_section = st.sidebar.radio("Go to", sections)


# Helper functions
@st.cache_data
def generate_ar_process(n_samples=500, ar_params=None, ma_params=None, d=0, random_seed=42):
    """Generate ARIMA process with specified parameters"""
    np.random.seed(random_seed)

    # Default to AR(1) if no params provided
    if ar_params is None:
        ar_params = [1]
    if ma_params is None:
        ma_params = [1]

    # Generate the base series
    series = arma_generate_sample(ar=ar_params, ma=ma_params, nsample=n_samples)

    # If d > 0, perform integration
    if d > 0:
        for _ in range(d):
            series = np.cumsum(series)

    return series


def adf_test_report(series):
    """Run ADF test and return formatted results"""
    result = adfuller(series)

    output = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        '1% Critical Value': result[4]['1%'],
        '5% Critical Value': result[4]['5%'],
        '10% Critical Value': result[4]['10%']
    }

    conclusion = ""
    if result[1] <= 0.05:
        conclusion = "Reject the null hypothesis. The series is stationary."
    else:
        conclusion = "Fail to reject the null hypothesis. The series has a unit root (non-stationary)."

    return output, conclusion


def kpss_test_report(series):
    """Run KPSS test and return formatted results"""
    result = kpss(series, regression='c')

    output = {
        'KPSS Statistic': result[0],
        'p-value': result[1],
        '1% Critical Value': result[3]['1%'],
        '5% Critical Value': result[3]['5%'],
        '10% Critical Value': result[3]['10%']
    }

    conclusion = ""
    if result[1] <= 0.05:
        conclusion = "Reject the null hypothesis. The series is non-stationary."
    else:
        conclusion = "Fail to reject the null hypothesis. The series is stationary."

    return output, conclusion


def plot_time_series(series, title, show_stats=True):
    """Plot time series with rolling statistics"""
    rolling_mean = pd.Series(series).rolling(window=20).mean()
    rolling_std = pd.Series(series).rolling(window=20).std()

    fig = go.Figure()

    # Add the series
    fig.add_trace(go.Scatter(
        y=series,
        mode='lines',
        name='Original Series',
        line=dict(color='blue')
    ))

    if show_stats:
        # Add rolling mean
        fig.add_trace(go.Scatter(
            y=rolling_mean,
            mode='lines',
            name='Rolling Mean (20)',
            line=dict(color='red', dash='dash')
        ))

        # Add rolling std
        fig.add_trace(go.Scatter(
            y=rolling_std,
            mode='lines',
            name='Rolling Std (20)',
            line=dict(color='green', dash='dot')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def add_structural_break(series, break_point, change_type='mean', mean_change=5, variance_multiplier=3):
    """Add a structural break to series"""
    new_series = series.copy()

    if change_type == 'mean':
        new_series[break_point:] = new_series[break_point:] + mean_change
    elif change_type == 'variance':
        new_series[break_point:] = new_series[break_point:] * variance_multiplier
    elif change_type == 'trend':
        slope = np.linspace(0, mean_change, len(new_series) - break_point)
        new_series[break_point:] = new_series[break_point:] + slope

    return new_series


# Introduction section
if selected_section == "Introduction to Stationarity":
    st.markdown('<div class="section-header">Introduction to Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    Stationarity is a fundamental concept in time series analysis. A stationary time series has statistical properties that do not change over time, which simplifies many statistical procedures.

    ### Why is Stationarity Important?

    - **Predictability**: Stationary series are more predictable and have well-defined statistical properties.
    - **Model Validity**: Many time series models (like ARIMA) assume stationarity.
    - **Statistical Inference**: Tests and confidence intervals are more reliable with stationary data.
    """)

    st.markdown('<div class="subsection-header">Definition of Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    A time series is stationary if:

    1. The **mean** of the series is constant over time
    2. The **variance** of the series is constant over time
    3. The **covariance** between observations depends only on the time lag between them, not on the actual time

    In simpler terms, a stationary series has no trend, constant variance, and doesn't show seasonal patterns.
    """)

    st.markdown(
        '<div class="info-box">A stationary process maintains the same statistical properties regardless of where in time we observe it.</div>',
        unsafe_allow_html=True)

    # Examples of stationary vs non-stationary processes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Stationary Process (White Noise)**")
        stationary_series = np.random.normal(0, 1, 500)
        st.plotly_chart(plot_time_series(stationary_series, "Stationary Series (White Noise)"),
                        use_container_width=True)

    with col2:
        st.markdown("**Non-stationary Process (Random Walk)**")
        non_stationary_series = np.cumsum(np.random.normal(0, 1, 500))
        st.plotly_chart(plot_time_series(non_stationary_series, "Non-stationary Series (Random Walk)"),
                        use_container_width=True)

# Types of Stationarity
elif selected_section == "Types of Stationarity":
    st.markdown('<div class="section-header">Types of Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    There are different degrees of stationarity that we consider in time series analysis:
    """)

    st.markdown('<div class="subsection-header">1. Strict Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    A time series is **strictly stationary** if the joint distribution of observations is invariant to time shifts. This means:

    - The complete probability distribution of the process does not change when shifted in time
    - All moments (mean, variance, kurtosis, etc.) remain constant over time
    - This is a very strong condition that is difficult to verify in practice
    """)

    st.markdown('<div class="subsection-header">2. Weak (Covariance) Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    A time series is **weakly stationary** (or covariance stationary) if:

    - The mean is constant over time: E[X_t] = μ for all t
    - The variance is constant over time: Var(X_t) = σ² for all t
    - The covariance between observations depends only on the lag: Cov(X_t, X_t+h) = γ(h) for all t and h

    This is the most common form of stationarity used in practice, and what most tests evaluate.
    """)

    st.markdown('<div class="subsection-header">3. Trend Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    A time series is **trend stationary** if:

    - After removing a deterministic trend, the series becomes stationary
    - The stochastic process driving the series is stationary, but around a trend
    - Can be transformed to stationary by detrending
    """)

    # Examples
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Weakly Stationary Series (AR(1) process)**")
        ar1_params = np.array([1, -0.7])  # AR(1) with coefficient 0.7
        ar1_series = arma_generate_sample(ar=ar1_params, ma=[1], nsample=500)
        st.plotly_chart(plot_time_series(ar1_series, "Weakly Stationary AR(1) Process"), use_container_width=True)

    with col2:
        st.markdown("**Trend Stationary Series**")
        # Create linear trend + stationary component
        t = np.arange(500)
        trend = 0.05 * t
        stationary_component = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=500)
        trend_stationary = trend + stationary_component
        st.plotly_chart(plot_time_series(trend_stationary, "Trend Stationary Series"), use_container_width=True)

    st.markdown('<div class="subsection-header">4. Difference Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    A time series is **difference stationary** if:

    - Taking the difference of the series makes it stationary
    - Contains a unit root (non-stationary)
    - Common example: random walk becomes stationary after differencing
    - The number of differences required to achieve stationarity is the order of integration
    """)

    # Example of difference stationarity
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Difference Stationary Series (Random Walk)**")
        random_walk = np.cumsum(np.random.normal(0, 1, 500))
        st.plotly_chart(plot_time_series(random_walk, "Random Walk (Non-stationary)"), use_container_width=True)

    with col2:
        st.markdown("**After First Differencing**")
        diff_series = np.diff(random_walk)
        st.plotly_chart(plot_time_series(diff_series, "First Difference (Stationary)"), use_container_width=True)

    st.markdown(
        '<div class="info-box">Understanding the type of stationarity helps in selecting the appropriate transformation method. Trend stationary series require detrending, while difference stationary series require differencing.</div>',
        unsafe_allow_html=True)

# Visualizing Stationarity
elif selected_section == "Visualizing Stationarity":
    st.markdown('<div class="section-header">Visualizing Stationarity</div>', unsafe_allow_html=True)

    st.markdown("""
    Visual inspection is often the first step in determining whether a time series is stationary. There are several graphical methods to assess stationarity:
    """)

    st.markdown('<div class="subsection-header">Time Series Plot with Rolling Statistics</div>', unsafe_allow_html=True)

    st.markdown("""
    One of the most common approaches is to plot the series along with its rolling mean and standard deviation:

    - If the rolling mean changes significantly over time → Non-stationary in mean
    - If the rolling standard deviation changes significantly → Non-stationary in variance
    """)

    # Create interactive example
    process_type = st.selectbox(
        "Select Process Type:",
        ["Stationary (White Noise)",
         "Stationary AR(1)",
         "Non-stationary (Random Walk)",
         "Trend Stationary",
         "Seasonally Non-stationary",
         "Heteroskedastic (Changing Variance)"]
    )

    window_size = st.slider("Rolling Window Size:", min_value=5, max_value=50, value=20)

    # Generate selected process
    n_samples = 500
    if process_type == "Stationary (White Noise)":
        series = np.random.normal(0, 1, n_samples)
        title = "White Noise Process (Stationary)"

    elif process_type == "Stationary AR(1)":
        ar1_params = np.array([1, -0.7])
        series = arma_generate_sample(ar=ar1_params, ma=[1], nsample=n_samples)
        title = "AR(1) Process with coefficient 0.7 (Stationary)"

    elif process_type == "Non-stationary (Random Walk)":
        series = np.cumsum(np.random.normal(0, 1, n_samples))
        title = "Random Walk Process (Non-stationary)"

    elif process_type == "Trend Stationary":
        t = np.arange(n_samples)
        trend = 0.05 * t
        stationary_component = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=n_samples)
        series = trend + stationary_component
        title = "Trend Stationary Process"

    elif process_type == "Seasonally Non-stationary":
        t = np.arange(n_samples)
        seasonal = 10 * np.sin(2 * np.pi * t / 50)
        noise = np.random.normal(0, 1, n_samples)
        series = seasonal + noise
        title = "Seasonally Non-stationary Process"

    elif process_type == "Heteroskedastic (Changing Variance)":
        series = np.random.normal(0, 1, n_samples)
        series[n_samples // 2:] = series[n_samples // 2:] * 3  # Increase variance in second half
        title = "Heteroskedastic Process (Non-stationary in variance)"

    # Calculate rolling statistics
    rolling_mean = pd.Series(series).rolling(window=window_size).mean()
    rolling_std = pd.Series(series).rolling(window=window_size).std()

    # Create plot
    fig = go.Figure()

    # Add the series
    fig.add_trace(go.Scatter(
        y=series,
        mode='lines',
        name='Original Series',
        line=dict(color='blue')
    ))

    # Add rolling mean
    fig.add_trace(go.Scatter(
        y=rolling_mean,
        mode='lines',
        name=f'Rolling Mean (window={window_size})',
        line=dict(color='red', dash='dash')
    ))

    # Add rolling std
    fig.add_trace(go.Scatter(
        y=rolling_std,
        mode='lines',
        name=f'Rolling Std (window={window_size})',
        line=dict(color='green', dash='dot')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation guidance based on process type
    if process_type == "Stationary (White Noise)":
        st.markdown(
            '<div class="info-box">In a white noise process, both rolling mean and standard deviation should be approximately constant. Notice how they fluctuate only slightly around fixed values.</div>',
            unsafe_allow_html=True)

    elif process_type == "Stationary AR(1)":
        st.markdown(
            '<div class="info-box">An AR(1) process with coefficient less than 1 is stationary. The rolling mean and standard deviation should stabilize after the initial periods.</div>',
            unsafe_allow_html=True)

    elif process_type == "Non-stationary (Random Walk)":
        st.markdown(
            '<div class="warning-box">The random walk process is non-stationary. Notice how the rolling mean changes significantly over time, following the "wandering" behavior of the series.</div>',
            unsafe_allow_html=True)

    elif process_type == "Trend Stationary":
        st.markdown(
            '<div class="warning-box">This series has a clear upward trend. The rolling mean also shows an upward trend, indicating non-stationarity in mean. However, differencing or detrending can make it stationary.</div>',
            unsafe_allow_html=True)

    elif process_type == "Seasonally Non-stationary":
        st.markdown(
            '<div class="warning-box">This series shows regular seasonal patterns. The rolling mean fluctuates in a cyclical pattern, indicating seasonal non-stationarity.</div>',
            unsafe_allow_html=True)

    elif process_type == "Heteroskedastic (Changing Variance)":
        st.markdown(
            '<div class="warning-box">This series has changing variance over time (heteroskedasticity). Notice how the rolling standard deviation increases significantly in the latter half of the series.</div>',
            unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">Other Visual Methods</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Autocorrelation Function (ACF) Plot

    The ACF plot shows the correlation between the series and its lagged values:

    - **Stationary series**: ACF drops to zero relatively quickly
    - **Non-stationary series**: ACF decays very slowly

    ### Partial Autocorrelation Function (PACF) Plot

    The PACF shows the correlation between the series and its lagged values, after removing the effects of intermediate lags:

    - Helps in identifying AR processes and determining the order
    """)

    # Show ACF and PACF for current series
    col1, col2 = st.columns(2)

    with col1:
        fig_acf = go.Figure()
        acf_values = sm.tsa.acf(series, nlags=40)
        fig_acf.add_trace(go.Scatter(
            x=list(range(len(acf_values))),
            y=acf_values,
            mode='markers+lines',
            name='ACF',
            line=dict(color='blue')
        ))

        # Add confidence intervals (approximately ±2/√n)
        conf_level = 1.96 / np.sqrt(len(series))
        fig_acf.add_trace(go.Scatter(
            x=list(range(len(acf_values))),
            y=[conf_level] * len(acf_values),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Upper CI'
        ))
        fig_acf.add_trace(go.Scatter(
            x=list(range(len(acf_values))),
            y=[-conf_level] * len(acf_values),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Lower CI'
        ))

        fig_acf.update_layout(
            title="Autocorrelation Function (ACF)",
            xaxis_title='Lag',
            yaxis_title='Correlation',
            height=400
        )
        st.plotly_chart(fig_acf, use_container_width=True)

    with col2:
        fig_pacf = go.Figure()
        pacf_values = sm.tsa.pacf(series, nlags=40)
        fig_pacf.add_trace(go.Scatter(
            x=list(range(len(pacf_values))),
            y=pacf_values,
            mode='markers+lines',
            name='PACF',
            line=dict(color='green')
        ))

        # Add confidence intervals
        fig_pacf.add_trace(go.Scatter(
            x=list(range(len(pacf_values))),
            y=[conf_level] * len(pacf_values),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Upper CI'
        ))
        fig_pacf.add_trace(go.Scatter(
            x=list(range(len(pacf_values))),
            y=[-conf_level] * len(pacf_values),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Lower CI'
        ))

        fig_pacf.update_layout(
            title="Partial Autocorrelation Function (PACF)",
            xaxis_title='Lag',
            yaxis_title='Partial Correlation',
            height=400
        )
        st.plotly_chart(fig_pacf, use_container_width=True)

    # ACF/PACF interpretation based on process
    if process_type == "Stationary (White Noise)":
        st.markdown(
            '<div class="info-box">For white noise, both ACF and PACF should show no significant correlations at any lag (except lag 0), as all values should be within the confidence intervals.</div>',
            unsafe_allow_html=True)

    elif process_type == "Stationary AR(1)":
        st.markdown(
            '<div class="info-box">For an AR(1) process, the ACF should decay exponentially, while the PACF should have a significant spike only at lag 1 and then drop to near zero.</div>',
            unsafe_allow_html=True)

    elif process_type == "Non-stationary (Random Walk)":
        st.markdown(
            '<div class="warning-box">For a random walk, the ACF decays very slowly, remaining significant at many lags - a strong indicator of non-stationarity.</div>',
            unsafe_allow_html=True)

    elif process_type == "Trend Stationary" or process_type == "Seasonally Non-stationary":
        st.markdown(
            '<div class="warning-box">For trend or seasonal series, the ACF typically shows a slow decay pattern, often with a periodic structure in the seasonal case.</div>',
            unsafe_allow_html=True)

    elif process_type == "Heteroskedastic (Changing Variance)":
        st.markdown(
            '<div class="info-box">Changing variance doesn\'t necessarily affect the ACF/PACF pattern directly, but it does affect the reliability of these measures.</div>',
            unsafe_allow_html=True)

# Unit Root Tests
elif selected_section == "Unit Root Tests":
    st.markdown('<div class="section-header">Unit Root Tests</div>', unsafe_allow_html=True)

    st.markdown("""
    Unit root tests are formal statistical procedures to determine whether a time series is stationary or not. The term "unit root" refers to a characteristic of non-stationary processes.
    """)

    st.markdown('<div class="subsection-header">Understanding Unit Roots</div>', unsafe_allow_html=True)

    st.markdown("""
    A unit root is a feature of a stochastic process that causes problems in statistical inference. Mathematically, if we have an AR(1) process:

    $Y_t = \phi Y_{t-1} + \epsilon_t$

    - If $|\phi| < 1$: The process is stationary
    - If $\phi = 1$: The process has a unit root (i.e., it's a random walk)
    - If $|\phi| > 1$: The process is explosive

    The presence of a unit root means the process is non-stationary, and shocks to the system have permanent effects.
    """)

    st.markdown(
        '<div class="info-box">When a time series has a unit root, it means the characteristic equation of the AR polynomial has a root equal to 1, which implies non-stationarity.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">Common Unit Root Tests</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Augmented Dickey-Fuller (ADF) Test

    The most widely used unit root test:

    - **Null hypothesis (H₀)**: The series has a unit root (non-stationary)
    - **Alternative hypothesis (H₁)**: The series is stationary
    - If p-value < significance level (e.g., 0.05), we reject H₀ and conclude the series is stationary

    ### 2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test

    Complements the ADF test with reversed hypotheses:

    - **Null hypothesis (H₀)**: The series is stationary
    - **Alternative hypothesis (H₁)**: The series has a unit root (non-stationary)
    - If p-value < significance level, we reject H₀ and conclude the series is non-stationary

    ### 3. Phillips-Perron (PP) Test

    Similar to ADF but with different handling of serial correlation:

    - **Null hypothesis (H₀)**: The series has a unit root
    - **Alternative hypothesis (H₁)**: The series is stationary
    - More robust to heteroskedasticity than ADF
    """)

    st.markdown('<div class="subsection-header">Comparison of Different Unit Root Test Results</div>',
                unsafe_allow_html=True)

    # Create interactive example for unit root testing
    test_series_type = st.selectbox(
        "Select Series Type to Test:",
        ["Stationary AR(1) (φ=0.7)",
         "Borderline Stationary AR(1) (φ=0.95)",
         "Unit Root Process (Random Walk)",
         "Explosive Process (φ=1.05)",
         "Trend Stationary Process"]
    )

    # Generate selected process
    n_samples = 200  # Smaller sample to show small sample issues
    np.random.seed(42)  # For reproducibility

    if test_series_type == "Stationary AR(1) (φ=0.7)":
        ar_params = np.array([1, -0.7])
        series = arma_generate_sample(ar=ar_params, ma=[1], nsample=n_samples)
        title = "Stationary AR(1) Process with φ=0.7"

    elif test_series_type == "Borderline Stationary AR(1) (φ=0.95)":
        ar_params = np.array([1, -0.95])
        series = arma_generate_sample(ar=ar_params, ma=[1], nsample=n_samples)
        title = "Near Unit Root AR(1) Process with φ=0.95"

    elif test_series_type == "Unit Root Process (Random Walk)":
        series = np.cumsum(np.random.normal(0, 1, n_samples))
        title = "Unit Root Process (Random Walk)"

    elif test_series_type == "Explosive Process (φ=1.05)":
        # For explosive process, we'll simulate manually to avoid numerical issues
        series = np.zeros(n_samples)
        series[0] = np.random.normal(0, 1)
        for t in range(1, n_samples):
            series[t] = 1.05 * series[t - 1] + np.random.normal(0, 1)
        title = "Explosive AR(1) Process with φ=1.05"

    elif test_series_type == "Trend Stationary Process":
        t = np.arange(n_samples)
        trend = 0.05 * t
        stationary_component = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=n_samples)
        series = trend + stationary_component
        title = "Trend Stationary Process"

    # Plot the series
    fig = plot_time_series(series, title)
    st.plotly_chart(fig, use_container_width=True)

    # Run tests
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="subsection-header">ADF Test Results</div>', unsafe_allow_html=True)
        adf_output, adf_conclusion = adf_test_report(series)

        for key, value in adf_output.items():
            st.write(f"**{key}:** {value:.4f}")

        st.markdown(f"**Conclusion:** {adf_conclusion}")

    with col2:
        st.markdown('<div class="subsection-header">KPSS Test Results</div>', unsafe_allow_html=True)
        kpss_output, kpss_conclusion = kpss_test_report(series)

        for key, value in kpss_output.items():
            st.write(f"**{key}:** {value:.4f}")

        st.markdown(f"**Conclusion:** {kpss_conclusion}")

    # Interpretation based on series type
    st.markdown('<div class="subsection-header">Interpretation</div>', unsafe_allow_html=True)

    if test_series_type == "Stationary AR(1) (φ=0.7)":
        st.markdown(
            '<div class="info-box">This is a stationary AR(1) process. We expect the ADF test to reject the null hypothesis (small p-value), indicating stationarity. The KPSS test should fail to reject the null (large p-value), also indicating stationarity.</div>',
            unsafe_allow_html=True)

    elif test_series_type == "Borderline Stationary AR(1) (φ=0.95)":
        st.markdown(
            '<div class="warning-box">This is a nearly non-stationary series with φ very close to 1. Tests may struggle to distinguish this from a true unit root process, especially with small samples. This illustrates the lower power of unit root tests near the unit circle.</div>',
            unsafe_allow_html=True)

    elif test_series_type == "Unit Root Process (Random Walk)":
        st.markdown(
            '<div class="info-box">This is a true unit root process (random walk). We expect the ADF test to fail to reject the null hypothesis (large p-value), indicating non-stationarity. The KPSS test should reject the null (small p-value), also indicating non-stationarity.</div>',
            unsafe_allow_html=True)

    elif test_series_type == "Explosive Process (φ=1.05)":
        st.markdown(
            '<div class="warning-box">This is an explosive process (φ > 1). The ADF test\'s null hypothesis is that φ = 1, and the alternative is φ < 1, so technically it\'s not designed for explosive processes. However, the extreme behavior usually leads to rejection of the null hypothesis, but not necessarily for the right reasons!</div>',
            unsafe_allow_html=True)

    elif test_series_type == "Trend Stationary Process":
        st.markdown(
            '<div class="warning-box">This is a trend stationary process. Without accounting for the trend, both tests might incorrectly classify it as non-stationary. The ADF test with trend specification would be more appropriate.</div>',
            unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">Comparing Test Results</div>', unsafe_allow_html=True)

    st.markdown("""
    When ADF and KPSS tests agree:

    - Both indicate stationary → Series is likely stationary
    - Both indicate non-stationary → Series is likely non-stationary

    When tests disagree:

    - ADF: Non-stationary, KPSS: Stationary → May have insufficient evidence, or the series could be fractionally integrated
    - ADF: Stationary, KPSS: Non-stationary → May indicate trend stationarity (consider testing with trend)
    """)

# Cases in Unit Root Tests
elif selected_section == "Cases in Unit Root Tests":
    st.markdown('<div class="section-header">Cases in Unit Root Tests</div>', unsafe_allow_html=True)

    st.markdown("""
    Unit root tests can include different components in their test equations, corresponding to different data generating processes. The most common specifications include:
    """)

    st.markdown('<div class="subsection-header">Test Equation Specifications</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. No Constant, No Trend (None)

    $\Delta y_t = \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$

    - Appropriate when the series fluctuates around zero
    - Rarely used in practice as most economic series have non-zero means

    ### 2. Constant, No Trend (Intercept)

    $\Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$

    - Most common specification
    - Appropriate when the series fluctuates around a non-zero constant
    - The null hypothesis allows for a random walk with drift

    ### 3. Constant and Trend (Trend)

    $\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$

    - Appropriate when the series appears to have a trend
    - Tests whether deviations from trend are stationary
    - The null hypothesis allows for a random walk with drift around a deterministic trend
    """)

    st.markdown(
        '<div class="info-box">Choosing the wrong specification can lead to incorrect conclusions about stationarity. If in doubt, it\'s often safest to include both constant and trend, as this specification is the most general.</div>',
        unsafe_allow_html=True)

    # Interactive example for different specifications
    st.markdown('<div class="subsection-header">Interactive Example: Effect of Test Specification</div>',
                unsafe_allow_html=True)

    case_series_type = st.selectbox(
        "Select Series Type:",
        ["Stationary around zero mean",
         "Stationary around non-zero mean",
         "Trend stationary",
         "Random walk",
         "Random walk with drift"]
    )

    # Generate selected process
    n_samples = 300
    np.random.seed(123)  # For reproducibility

    if case_series_type == "Stationary around zero mean":
        series = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=n_samples)
        title = "Stationary Series around Zero Mean"

    elif case_series_type == "Stationary around non-zero mean":
        series = 5 + arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=n_samples)
        title = "Stationary Series around Non-zero Mean (μ=5)"

    elif case_series_type == "Trend stationary":
        t = np.arange(n_samples)
        trend = 0.05 * t
        stationary_component = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=n_samples)
        series = trend + stationary_component
        title = "Trend Stationary Series"

    elif case_series_type == "Random walk":
        series = np.cumsum(np.random.normal(0, 1, n_samples))
        title = "Random Walk (No Drift)"

    elif case_series_type == "Random walk with drift":
        series = np.cumsum(0.1 + np.random.normal(0, 1, n_samples))
        title = "Random Walk with Drift (μ=0.1)"

    # Plot the series
    fig = plot_time_series(series, title)
    st.plotly_chart(fig, use_container_width=True)

    # Run ADF test with different specifications
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="subsection-header">No Constant, No Trend</div>', unsafe_allow_html=True)
        result_none = adfuller(series, regression='nc')

        st.write(f"**ADF Statistic:** {result_none[0]:.4f}")
        st.write(f"**p-value:** {result_none[1]:.4f}")
        st.write(f"**1% Critical Value:** {result_none[4]['1%']:.4f}")
        st.write(f"**5% Critical Value:** {result_none[4]['5%']:.4f}")

        if result_none[1] <= 0.05:
            st.write("**Conclusion:** Reject H₀ (Stationary)")
        else:
            st.write("**Conclusion:** Fail to reject H₀ (Non-stationary)")

    with col2:
        st.markdown('<div class="subsection-header">With Constant</div>', unsafe_allow_html=True)
        result_c = adfuller(series, regression='c')

        st.write(f"**ADF Statistic:** {result_c[0]:.4f}")
        st.write(f"**p-value:** {result_c[1]:.4f}")
        st.write(f"**1% Critical Value:** {result_c[4]['1%']:.4f}")
        st.write(f"**5% Critical Value:** {result_c[4]['5%']:.4f}")

        if result_c[1] <= 0.05:
            st.write("**Conclusion:** Reject H₀ (Stationary)")
        else:
            st.write("**Conclusion:** Fail to reject H₀ (Non-stationary)")

    with col3:
        st.markdown('<div class="subsection-header">With Constant and Trend</div>', unsafe_allow_html=True)
        result_ct = adfuller(series, regression='ct')

        st.write(f"**ADF Statistic:** {result_ct[0]:.4f}")
        st.write(f"**p-value:** {result_ct[1]:.4f}")
        st.write(f"**1% Critical Value:** {result_ct[4]['1%']:.4f}")
        st.write(f"**5% Critical Value:** {result_ct[4]['5%']:.4f}")

        if result_ct[1] <= 0.05:
            st.write("**Conclusion:** Reject H₀ (Stationary)")
        else:
            st.write("**Conclusion:** Fail to reject H₀ (Non-stationary)")

    # Interpretation based on series type
    st.markdown('<div class="subsection-header">Interpretation</div>', unsafe_allow_html=True)

    if case_series_type == "Stationary around zero mean":
        st.markdown(
            '<div class="info-box">For a stationary series with zero mean, all three specifications should correctly identify it as stationary. However, adding unnecessary components (constant/trend) reduces power.</div>',
            unsafe_allow_html=True)

    elif case_series_type == "Stationary around non-zero mean":
        st.markdown(
            '<div class="warning-box">For a stationary series with non-zero mean, the "no constant" specification is misspecified and might incorrectly suggest non-stationarity. The constant and trend specifications should correctly identify it as stationary.</div>',
            unsafe_allow_html=True)

    elif case_series_type == "Trend stationary":
        st.markdown(
            '<div class="warning-box">For a trend stationary series, only the specification with trend is correct. Without accounting for the trend, the other specifications might incorrectly suggest non-stationarity.</div>',
            unsafe_allow_html=True)

    elif case_series_type == "Random walk":
        st.markdown(
            '<div class="info-box">For a pure random walk, all specifications should indicate non-stationarity. However, including unnecessary components can reduce power, especially in small samples.</div>',
            unsafe_allow_html=True)

    elif case_series_type == "Random walk with drift":
        st.markdown(
            '<div class="warning-box">For a random walk with drift, all specifications should indicate non-stationarity. However, the specification with trend might have lower power to detect the unit root due to the drift mimicking a deterministic trend.</div>',
            unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">Guidelines for Choosing the Specification</div>',
                unsafe_allow_html=True)

    st.markdown("""
    1. **Visual inspection first**: Look at the plot of your data to see if there's a clear trend or non-zero mean

    2. **Start general, then simplify**:
        - Begin with the most general specification (constant and trend)
        - If the trend is not significant, drop it and retest with only a constant
        - If the constant is not significant, drop it and retest with no constant

    3. **Consider theoretical knowledge**: Some economic series are known to have trends or non-zero means by nature

    4. **When in doubt**: Using the constant-only specification is often a safe middle ground
    """)

# Structural Breaks
elif selected_section == "Structural Breaks":
    st.markdown('<div class="section-header">Structural Breaks</div>', unsafe_allow_html=True)

    st.markdown("""
    Structural breaks are sudden changes in the parameters of a time series model, which can significantly affect the stationarity properties and the performance of unit root tests.
    """)

    st.markdown('<div class="subsection-header">Types of Structural Breaks</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Mean Shift (Level Break)

    A sudden change in the mean level of the series:

    $y_t = \mu_1 + \epsilon_t$ for $t \leq T_b$

    $y_t = \mu_2 + \epsilon_t$ for $t > T_b$

    - Where $T_b$ is the break point and $\mu_1 \neq \mu_2$
    - Common in series affected by policy changes or economic events

    ### 2. Trend Break

    A change in the slope of the trend:

    $y_t = \alpha + \beta_1 t + \epsilon_t$ for $t \leq T_b$

    $y_t = \alpha + \beta_1 T_b + \beta_2 (t - T_b) + \epsilon_t$ for $t > T_b$

    - Where $\beta_1 \neq \beta_2$
    - Often seen in economic growth series after structural reforms

    ### 3. Variance Break

    A change in the volatility of the series:

    $y_t = \mu + \epsilon_t$ where $\epsilon_t \sim N(0, \sigma_1^2)$ for $t \leq T_b$

    $y_t = \mu + \epsilon_t$ where $\epsilon_t \sim N(0, \sigma_2^2)$ for $t > T_b$

    - Where $\sigma_1^2 \neq \sigma_2^2$
    - Common in financial time series after market regime changes

    ### 4. Multiple Breaks

    Series may contain multiple structural breaks of different types
    """)

    st.markdown('<div class="subsection-header">Impact on Unit Root Tests</div>', unsafe_allow_html=True)

    st.markdown("""
    Structural breaks can severely bias standard unit root tests:

    - **Reduced Power**: Tests are less likely to reject the null hypothesis of a unit root even when the series is actually stationary around a broken trend or mean
    - **False Non-rejection**: A stationary series with structural breaks may be incorrectly classified as non-stationary
    - **Size Distortions**: The actual rejection rate under the null hypothesis can be very different from the nominal significance level
    """)

    st.markdown(
        '<div class="warning-box">Perron (1989) showed that the presence of a structural break reduces the power of standard ADF tests, often leading to false conclusions about non-stationarity.</div>',
        unsafe_allow_html=True)

    # Interactive demonstration of structural breaks
    st.markdown('<div class="subsection-header">Interactive Demonstration: Impact of Structural Breaks</div>',
                unsafe_allow_html=True)

    break_type = st.selectbox(
        "Select Type of Structural Break:",
        ["Mean Shift (Level Break)",
         "Trend Break",
         "Variance Break",
         "Multiple Breaks"]
    )

    break_point = st.slider("Break Point (as % of sample):",
                            min_value=10, max_value=90, value=50)

    # Generate base series (stationary AR(1))
    n_samples = 400
    np.random.seed(456)

    # Convert percentage to index
    break_idx = int(n_samples * break_point / 100)

    # Base stationary series
    base_series = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=n_samples)

    # Add structural break based on selected type
    if break_type == "Mean Shift (Level Break)":
        mean_shift = 3.0
        series_with_break = add_structural_break(base_series, break_idx, 'mean', mean_shift)
        title = f"Stationary Series with Mean Shift at t={break_idx}"
        break_description = f"Mean increases by {mean_shift} units at the break point"

    elif break_type == "Trend Break":
        trend_change = 0.02
        # Add linear trend after break point
        series_with_break = base_series.copy()
        for i in range(break_idx, n_samples):
            series_with_break[i] += trend_change * (i - break_idx)
        title = f"Stationary Series with Trend Break at t={break_idx}"
        break_description = f"Trend slope changes to {trend_change} after the break point"

    elif break_type == "Variance Break":
        variance_multiplier = 3.0
        series_with_break = add_structural_break(base_series, break_idx, 'variance',
                                                 variance_multiplier=variance_multiplier)
        title = f"Stationary Series with Variance Break at t={break_idx}"
        break_description = f"Variance increases by a factor of {variance_multiplier} after the break point"

    elif break_type == "Multiple Breaks":
        # First break at 1/3 (mean shift)
        break_idx1 = n_samples // 3
        # Second break at 2/3 (variance change)
        break_idx2 = 2 * n_samples // 3

        series_with_break = add_structural_break(base_series, break_idx1, 'mean', mean_change=2.0)
        series_with_break = add_structural_break(series_with_break, break_idx2, 'variance', variance_multiplier=2.5)

        title = f"Stationary Series with Multiple Breaks at t={break_idx1} and t={break_idx2}"
        break_description = f"Mean shift at t={break_idx1} and variance increase at t={break_idx2}"

    # Plot original and series with break
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Stationary Series**")
        fig1 = plot_time_series(base_series, "Stationary AR(1) Process")
        st.plotly_chart(fig1, use_container_width=True)

        # Run ADF test on original series
        result_orig = adfuller(base_series)
        st.write(f"**ADF Statistic:** {result_orig[0]:.4f}")
        st.write(f"**p-value:** {result_orig[1]:.4f}")

        if result_orig[1] <= 0.05:
            st.write("**Conclusion:** Reject H₀ (Stationary)")
        else:
            st.write("**Conclusion:** Fail to reject H₀ (Non-stationary)")

    with col2:
        st.markdown(f"**Series with {break_type}**")
        fig2 = plot_time_series(series_with_break, title)
        st.plotly_chart(fig2, use_container_width=True)

        # Run ADF test on series with break
        result_break = adfuller(series_with_break)
        st.write(f"**ADF Statistic:** {result_break[0]:.4f}")
        st.write(f"**p-value:** {result_break[1]:.4f}")

        if result_break[1] <= 0.05:
            st.write("**Conclusion:** Reject H₀ (Stationary)")
        else:
            st.write("**Conclusion:** Fail to reject H₀ (Non-stationary)")

    st.markdown(f"**Break Description:** {break_description}")

    st.markdown(
        '<div class="warning-box">Notice how the ADF test results can change dramatically when a structural break is present, even though the underlying process (excluding the break) is stationary. This demonstrates how structural breaks can lead to incorrect conclusions about stationarity.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">Unit Root Tests with Structural Breaks</div>', unsafe_allow_html=True)

    st.markdown("""
    Several specialized tests have been developed to handle structural breaks:

    ### 1. Perron (1989) Test

    - Assumes the break point is known
    - Modifies the ADF test equation to include dummy variables for the break
    - Three variants: change in level, change in slope, or both

    ### 2. Zivot-Andrews (1992) Test

    - Determines the break point endogenously
    - Tests the null of a unit root against the alternative of stationarity with a break
    - More practical when the break point is unknown

    ### 3. Clemente-Montañés-Reyes Test

    - Can account for one or two structural breaks
    - Offers Additive Outlier (AO) and Innovative Outlier (IO) models
    - AO allows for sudden changes, IO for gradual changes

    ### 4. Lee-Strazicich Test

    - Tests the null of a unit root with breaks against the alternative of stationarity with breaks
    - Can handle one or two breaks in level and/or trend
    """)

    st.markdown(
        '<div class="info-box">When structural breaks are suspected, it\'s advisable to use specialized tests like Zivot-Andrews or Lee-Strazicich rather than standard unit root tests to avoid misleading conclusions.</div>',
        unsafe_allow_html=True)

# Power of Unit Root Tests
elif selected_section == "Power of Unit Root Tests":
    st.markdown('<div class="section-header">Power of Unit Root Tests</div>', unsafe_allow_html=True)

    st.markdown("""
    The power of a statistical test is its ability to correctly reject the null hypothesis when it is false. For unit root tests, this means correctly identifying a stationary series as stationary.
    """)

    st.markdown('<div class="subsection-header">Factors Affecting Power</div>', unsafe_allow_html=True)

    st.markdown("""
    Several factors can significantly affect the power of unit root tests:

    ### 1. Sample Size

    - Smaller samples reduce test power
    - With small samples, tests often fail to reject the null of a unit root even when the series is stationary
    - Asymptotic critical values become less reliable

    ### 2. Proximity to Unit Root

    - Near-integrated processes (e.g., AR(1) with φ = 0.95) are difficult to distinguish from true unit root processes
    - Power decreases as the autoregressive parameter approaches 1

    ### 3. Deterministic Components

    - Including unnecessary constants or trends reduces power
    - Omitting necessary components can lead to size distortions

    ### 4. Structural Breaks

    - Unaccounted structural breaks severely reduce power
    - Standard tests tend to find unit roots in stationary series with breaks

    ### 5. Moving Average Components

    - Negative MA components can cause size distortions
    - Tests may over-reject or under-reject the null hypothesis
    """)

    # Interactive demonstration of power issues
    st.markdown('<div class="subsection-header">Interactive Demonstration: Power Issues</div>', unsafe_allow_html=True)

    power_issue = st.selectbox(
        "Select Power Issue to Demonstrate:",
        ["Sample Size Effect",
         "Near Unit Root Process",
         "Structural Break Effect"]
    )

    # Set seed for reproducibility
    np.random.seed(789)

    if power_issue == "Sample Size Effect":
        # Allow user to select sample size
        sample_size = st.select_slider(
            "Select Sample Size:",
            options=[30, 50, 100, 200, 500, 1000]
        )

        # Generate stationary AR(1) process
        ar_params = np.array([1, -0.8])
        series = arma_generate_sample(ar=ar_params, ma=[1], nsample=sample_size)

        title = f"Stationary AR(1) with φ=0.8 (Sample Size = {sample_size})"
        issue_description = "As sample size decreases, it becomes harder for the test to correctly identify a stationary series"

    elif power_issue == "Near Unit Root Process":
        # Allow user to select AR coefficient
        ar_coef = st.slider(
            "Select AR Coefficient (φ):",
            min_value=0.5, max_value=0.99, value=0.9, step=0.05
        )

        # Generate near unit root process
        ar_params = np.array([1, -ar_coef])
        series = arma_generate_sample(ar=ar_params, ma=[1], nsample=200)

        title = f"Near Unit Root Process with φ={ar_coef} (Sample Size = 200)"
        issue_description = "As φ approaches 1, it becomes harder to distinguish from a true unit root"

    elif power_issue == "Structural Break Effect":
        # Generate stationary series with break
        sample_size = 200
        break_point = sample_size // 2

        base_series = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=sample_size)
        series = add_structural_break(base_series, break_point, 'mean', mean_change=3.0)

        title = f"Stationary Series with Mean Break at t={break_point}"
        issue_description = "Structural breaks can make stationary series appear non-stationary to standard tests"

    # Plot the series
    fig = plot_time_series(series, title)
    st.plotly_chart(fig, use_container_width=True)

    # Run ADF test
    result = adfuller(series)

    st.write(f"**ADF Statistic:** {result[0]:.4f}")
    st.write(f"**p-value:** {result[1]:.4f}")
    st.write(f"**1% Critical Value:** {result[4]['1%']:.4f}")
    st.write(f"**5% Critical Value:** {result[4]['5%']:.4f}")

    if result[1] <= 0.05:
        test_conclusion = "Reject H₀ (Correctly identified as stationary)"
    else:
        test_conclusion = "Fail to reject H₀ (Incorrectly classified as non-stationary)"

    st.write(f"**Test Conclusion:** {test_conclusion}")
    st.markdown(f"**Issue Description:** {issue_description}")

    st.markdown('<div class="subsection-header">Power Curves</div>', unsafe_allow_html=True)

    st.markdown("""
    Power curves show how the probability of correctly rejecting the null hypothesis varies with different parameter values. For unit root tests, power typically:

    - Increases with sample size
    - Decreases as the AR coefficient approaches 1
    - Is affected by the inclusion of deterministic terms (constant, trend)

    As an example, the power of the ADF test might be only around 0.50 for a sample size of 100 and an AR coefficient of 0.90, meaning it would correctly identify the series as stationary only 50% of the time.
    """)

    # Theoretical power curve visualization
    ar_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    # Approximate power values based on Monte Carlo studies
    # These are simplified and for illustration purposes
    power_n50 = [0.99, 0.95, 0.85, 0.65, 0.30, 0.15, 0.07]
    power_n100 = [1.00, 0.99, 0.95, 0.80, 0.50, 0.25, 0.10]
    power_n500 = [1.00, 1.00, 1.00, 0.99, 0.90, 0.70, 0.25]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ar_values,
        y=power_n50,
        mode='lines+markers',
        name='n = 50',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=ar_values,
        y=power_n100,
        mode='lines+markers',
        name='n = 100',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=ar_values,
        y=power_n500,
        mode='lines+markers',
        name='n = 500',
        line=dict(color='red')
    ))

    # Add reference line at 0.05 significance level
    fig.add_trace(go.Scatter(
        x=[0.5, 0.99],
        y=[0.05, 0.05],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='α = 0.05'
    ))

    fig.update_layout(
        title="Approximate Power Curves for ADF Test",
        xaxis_title='AR Coefficient (φ)',
        yaxis_title='Power (Probability of Rejecting H₀)',
        xaxis=dict(tickvals=ar_values),
        yaxis=dict(range=[0, 1.05]),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="info-box">The power curves show how the ability to correctly identify a stationary series decreases as the AR coefficient approaches 1, and increases with larger sample sizes. For near unit root processes (φ > 0.9), even with moderately large samples, the power can be quite low.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">Strategies to Address Power Issues</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Use Multiple Tests

    - Combine different tests (ADF, KPSS, PP) for more robust conclusions
    - Tests with different null hypotheses can provide complementary evidence

    ### 2. Consider Panel Tests

    - When multiple related series are available, panel unit root tests can have higher power
    - Examples: Im-Pesaran-Shin (IPS), Levin-Lin-Chu (LLC)

    ### 3. Use Specialized Tests

    - For structural breaks: Zivot-Andrews, Lee-Strazicich
    - For seasonal data: HEGY test

    ### 4. Increase Lag Order Carefully

    - Too few lags: size distortions
    - Too many lags: reduced power
    - Use information criteria (AIC, BIC) to select optimal lag length

    ### 5. Adjust Critical Values

    - Small sample critical values are more appropriate than asymptotic ones
    - Some tests offer finite sample critical values
    """)

    st.markdown(
        '<div class="warning-box">Given the limitations of unit root tests, especially in small samples, conclusions about stationarity should not rely solely on statistical tests. Economic theory, visual inspection, and multiple testing approaches should inform the final decision.</div>',
        unsafe_allow_html=True)

# Interactive Examples
elif selected_section == "Interactive Examples":
    st.markdown('<div class="section-header">Interactive Examples & Exercises</div>', unsafe_allow_html=True)

    st.markdown("""
    This section provides hands-on examples and exercises to consolidate your understanding of stationarity concepts and unit root testing.
    """)

    example_type = st.radio(
        "Select Example Type:",
        ["Time Series Generator",
         "Transformation Explorer",
         "Test Your Knowledge"]
    )

    if example_type == "Time Series Generator":
        st.markdown('<div class="subsection-header">Time Series Generator</div>', unsafe_allow_html=True)

        st.markdown("""
        Generate different types of time series and analyze their stationarity properties.
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Process selection
            process_type = st.selectbox(
                "Select Process Type:",
                ["White Noise",
                 "AR(1)",
                 "MA(1)",
                 "ARMA(1,1)",
                 "Random Walk",
                 "Random Walk with Drift",
                 "Trend + Noise",
                 "Seasonal"]
            )

            # Sample size
            n_samples = st.slider("Sample Size:", min_value=100, max_value=1000, value=500, step=100)

        with col2:
            # Parameters based on process type
            if process_type == "AR(1)":
                ar_coef = st.slider("AR Coefficient (φ):", min_value=-0.99, max_value=0.99, value=0.7, step=0.05)
                noise_sd = st.slider("Error Standard Deviation:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            elif process_type == "MA(1)":
                ma_coef = st.slider("MA Coefficient (θ):", min_value=-0.99, max_value=0.99, value=0.7, step=0.05)
                noise_sd = st.slider("Error Standard Deviation:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            elif process_type == "ARMA(1,1)":
                ar_coef = st.slider("AR Coefficient (φ):", min_value=-0.99, max_value=0.99, value=0.7, step=0.05)
                ma_coef = st.slider("MA Coefficient (θ):", min_value=-0.99, max_value=0.99, value=0.3, step=0.05)
                noise_sd = 1.0

            elif process_type == "Random Walk with Drift":
                drift = st.slider("Drift Term:", min_value=-0.5, max_value=0.5, value=0.1, step=0.05)
                noise_sd = st.slider("Innovation Standard Deviation:", min_value=0.1, max_value=5.0, value=1.0,
                                     step=0.1)

            elif process_type == "Trend + Noise":
                trend_coef = st.slider("Trend Coefficient:", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
                noise_sd = st.slider("Noise Standard Deviation:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            elif process_type == "Seasonal":
                season_period = st.slider("Seasonal Period:", min_value=4, max_value=24, value=12, step=1)
                season_amplitude = st.slider("Seasonal Amplitude:", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
                noise_sd = st.slider("Noise Standard Deviation:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            else:  # White Noise or Random Walk
                noise_sd = st.slider("Standard Deviation:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        # Generate selected process
        np.random.seed(42)  # For reproducibility

        if process_type == "White Noise":
            series = np.random.normal(0, noise_sd, n_samples)
            title = f"White Noise (σ={noise_sd})"
            expected_stationary = True

        elif process_type == "AR(1)":
            ar_params = np.array([1, -ar_coef])
            series = arma_generate_sample(ar=ar_params, ma=[1], nsample=n_samples, scale=noise_sd)
            title = f"AR(1) Process with φ={ar_coef}"
            expected_stationary = abs(ar_coef) < 1

        elif process_type == "MA(1)":
            ma_params = np.array([1, ma_coef])
            series = arma_generate_sample(ar=[1], ma=ma_params, nsample=n_samples, scale=noise_sd)
            title = f"MA(1) Process with θ={ma_coef}"
            expected_stationary = True

        elif process_type == "ARMA(1,1)":
            ar_params = np.array([1, -ar_coef])
            ma_params = np.array([1, ma_coef])
            series = arma_generate_sample(ar=ar_params, ma=ma_params, nsample=n_samples)
            title = f"ARMA(1,1) Process with φ={ar_coef}, θ={ma_coef}"
            expected_stationary = abs(ar_coef) < 1

        elif process_type == "Random Walk":
            series = np.cumsum(np.random.normal(0, noise_sd, n_samples))
            title = "Random Walk Process"
            expected_stationary = False

        elif process_type == "Random Walk with Drift":
            innovations = drift + np.random.normal(0, noise_sd, n_samples)
            series = np.cumsum(innovations)
            title = f"Random Walk with Drift={drift}"
            expected_stationary = False

        elif process_type == "Trend + Noise":
            t = np.arange(n_samples)
            trend = trend_coef * t
            noise = np.random.normal(0, noise_sd, n_samples)
            series = trend + noise
            title = f"Trend (β={trend_coef}) + Noise"
            expected_stationary = False

        elif process_type == "Seasonal":
            t = np.arange(n_samples)
            seasonal = season_amplitude * np.sin(2 * np.pi * t / season_period)
            noise = np.random.normal(0, noise_sd, n_samples)
            series = seasonal + noise
            title = f"Seasonal Process (Period={season_period})"
            expected_stationary = False

        # Plot the series with rolling statistics
        fig = plot_time_series(series, title)
        st.plotly_chart(fig, use_container_width=True)

        # Run tests
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="subsection-header">ADF Test Results</div>', unsafe_allow_html=True)
            adf_output, adf_conclusion = adf_test_report(series)

            for key, value in adf_output.items():
                st.write(f"**{key}:** {value:.4f}")

            st.markdown(f"**Conclusion:** {adf_conclusion}")

        with col2:
            st.markdown('<div class="subsection-header">KPSS Test Results</div>', unsafe_allow_html=True)
            kpss_output, kpss_conclusion = kpss_test_report(series)

            for key, value in kpss_output.items():
                st.write(f"**{key}:** {value:.4f}")

            st.markdown(f"**Conclusion:** {kpss_conclusion}")

        # Theoretical properties
        st.markdown('<div class="subsection-header">Theoretical Properties</div>', unsafe_allow_html=True)

        st.markdown(f"""
        **Process Type:** {process_type}

        **Expected Stationarity:** {"Stationary" if expected_stationary else "Non-stationary"}

        **Explanation:**
        """)

        if process_type == "White Noise":
            st.markdown(
                "White noise is a classical example of a stationary process. It has constant mean (0), constant variance, and no autocorrelation between observations.")

        elif process_type == "AR(1)":
            if abs(ar_coef) < 1:
                st.markdown(
                    f"AR(1) process with |φ| < 1 (in this case, |{ar_coef}| < 1) is stationary. The process reverts to its mean.")
            else:
                st.markdown(
                    f"AR(1) process with |φ| ≥ 1 (in this case, |{ar_coef}| ≥ 1) is non-stationary. The process does not revert to a mean value.")

        elif process_type == "MA(1)":
            st.markdown(
                "MA processes of any order are always stationary (assuming finite variance of innovations), as they are finite linear combinations of white noise.")

        elif process_type == "ARMA(1,1)":
            if abs(ar_coef) < 1:
                st.markdown(
                    f"ARMA(1,1) with |φ| < 1 (in this case, |{ar_coef}| < 1) is stationary. The stationarity depends only on the AR component.")
            else:
                st.markdown(f"ARMA(1,1) with |φ| ≥ 1 (in this case, |{ar_coef}| ≥ 1) is non-stationary.")

        elif process_type == "Random Walk":
            st.markdown(
                "Random walk is the classic example of a non-stationary process. It has a unit root (φ=1 in an AR(1)), and shocks have permanent effects.")

        elif process_type == "Random Walk with Drift":
            st.markdown(
                "Random walk with drift is non-stationary. In addition to the unit root, it has a deterministic trend component (drift).")

        elif process_type == "Trend + Noise":
            st.markdown(
                "This process is trend non-stationary. However, it is stationary around a deterministic trend (trend-stationary).")

        elif process_type == "Seasonal":
            st.markdown(
                "This process exhibits seasonal non-stationarity due to the periodic component. The mean changes systematically with the season.")

    elif example_type == "Transformation Explorer":
        st.markdown('<div class="subsection-header">Transformation Explorer</div>', unsafe_allow_html=True)

        st.markdown("""
        Explore how different transformations can convert non-stationary time series to stationary ones.
        """)

        # Select non-stationary process
        non_stationary_type = st.selectbox(
            "Select Non-stationary Process:",
            ["Random Walk",
             "Random Walk with Drift",
             "Trend + Noise",
             "Seasonal Pattern",
             "Exponential Growth"]
        )

        # Generate selected non-stationary process
        np.random.seed(123)
        n_samples = 400

        if non_stationary_type == "Random Walk":
            original_series = np.cumsum(np.random.normal(0, 1, n_samples))
            title = "Random Walk Process"

        elif non_stationary_type == "Random Walk with Drift":
            innovations = 0.1 + np.random.normal(0, 1, n_samples)
            original_series = np.cumsum(innovations)
            title = "Random Walk with Drift (μ=0.1)"

        elif non_stationary_type == "Trend + Noise":
            t = np.arange(n_samples)
            trend = 0.05 * t
            noise = np.random.normal(0, 1, n_samples)
            original_series = trend + noise
            title = "Linear Trend + Noise"


        elif non_stationary_type == "Seasonal Pattern":

            t = np.arange(n_samples)

            seasonal = 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, n_samples)

            # Add upward trend

            trend = 0.02 * t

            original_series = seasonal + trend

            title = "Seasonal Pattern with Trend"


        elif non_stationary_type == "Exponential Growth":
            t = np.arange(n_samples)
            growth = np.exp(0.01 * t)
            noise_factor = 0.05  # Proportional noise
            noise = growth * noise_factor * np.random.normal(0, 1, n_samples)
            original_series = growth + noise
            title = "Exponential Growth"

        # Plot original series
        st.markdown('<div class="subsection-header">Original Non-stationary Series</div>', unsafe_allow_html=True)
        fig_orig = plot_time_series(original_series, title)
        st.plotly_chart(fig_orig, use_container_width=True)

        # Test original series
        adf_orig, adf_orig_concl = adf_test_report(original_series)
        st.write(f"**ADF p-value:** {adf_orig['p-value']:.4f} - {adf_orig_concl}")

        # Select transformation
        transformation = st.selectbox(
            "Select Transformation:",
            ["First Difference",
             "Second Difference",
             "Log Transform",
             "Log Difference (Growth Rate)",
             "Remove Trend",
             "Seasonal Difference",
             "HP Filter"]
        )

        # Apply selected transformation
        if transformation == "First Difference":
            transformed_series = np.diff(original_series)
            transform_title = "First Difference"
            transform_description = "Taking the first difference removes a unit root or a linear trend"

        elif transformation == "Second Difference":
            transformed_series = np.diff(np.diff(original_series))
            transform_title = "Second Difference"
            transform_description = "Taking the second difference can remove quadratic trends or processes integrated of order 2"

        elif transformation == "Log Transform":
            # Ensure positive values for log
            min_val = np.min(original_series)
            if min_val <= 0:
                adjusted_series = original_series - min_val + 1
            else:
                adjusted_series = original_series
            transformed_series = np.log(adjusted_series)
            transform_title = "Log Transform"
            transform_description = "Log transformation can stabilize variance, especially for series with exponential growth"

        elif transformation == "Log Difference (Growth Rate)":
            # Ensure positive values for log
            min_val = np.min(original_series)
            if min_val <= 0:
                adjusted_series = original_series - min_val + 1
            else:
                adjusted_series = original_series
            log_series = np.log(adjusted_series)
            transformed_series = np.diff(log_series)
            transform_title = "Log Difference (Growth Rate)"
            transform_description = "Log differences represent growth rates and can make exponential growth series stationary"

        elif transformation == "Remove Trend":
            # Simple linear detrending
            t = np.arange(n_samples)
            model = np.polyfit(t, original_series, 1)
            trend = model[0] * t + model[1]
            transformed_series = original_series - trend
            transform_title = "Detrended Series"
            transform_description = "Removing a fitted trend can make trend-stationary series stationary"

        elif transformation == "Seasonal Difference":
            # Use seasonal lag of 12 (common for monthly data)
            season_lag = 12
            transformed_series = original_series[season_lag:] - original_series[:-season_lag]
            transform_title = f"Seasonal Difference (lag={season_lag})"
            transform_description = "Seasonal differencing removes seasonal patterns of a fixed period"

        elif transformation == "HP Filter":
            # Simplified HP filter - let's use a polynomial approximation for illustration
            t = np.arange(n_samples)
            model = np.polyfit(t, original_series, 3)  # Cubic trend
            trend = np.polyval(model, t)
            transformed_series = original_series - trend
            transform_title = "HP Filter (Trend Removal)"
            transform_description = "The HP filter separates a time series into trend and cyclical components"

        # Plot transformed series
        st.markdown(f'<div class="subsection-header">Transformed Series: {transform_title}</div>',
                    unsafe_allow_html=True)
        fig_trans = plot_time_series(transformed_series, transform_title)
        st.plotly_chart(fig_trans, use_container_width=True)

        # Test transformed series
        adf_trans, adf_trans_concl = adf_test_report(transformed_series)
        st.write(f"**ADF p-value:** {adf_trans['p-value']:.4f} - {adf_trans_concl}")

        st.markdown(f"**Transformation Description:** {transform_description}")

        # Additional notes on transformation
        st.markdown('<div class="subsection-header">Transformation Guidelines</div>', unsafe_allow_html=True)

        if non_stationary_type == "Random Walk":
            st.markdown("""
            **For Random Walks:**
            - First differencing is typically sufficient
            - The resulting series should resemble white noise
            """)

        elif non_stationary_type == "Random Walk with Drift":
            st.markdown("""
            **For Random Walks with Drift:**
            - First differencing removes both the unit root and the drift
            - The resulting series should have a constant mean
            """)

        elif non_stationary_type == "Trend + Noise":
            st.markdown("""
            **For Trend + Noise:**
            - Detrending directly removes the deterministic trend
            - First differencing also works but may introduce MA correlations
            """)

        elif non_stationary_type == "Seasonal Pattern":
            st.markdown("""
            **For Seasonal Patterns:**
            - Seasonal differencing removes the seasonal component
            - Combined regular and seasonal differencing may be needed for series with both trend and seasonality
            """)

        elif non_stationary_type == "Exponential Growth":
            st.markdown("""
            **For Exponential Growth:**
            - Log transformation stabilizes variance
            - Log differencing gives growth rates, which are often stationary
            """)

    elif example_type == "Test Your Knowledge":
        st.markdown('<div class="subsection-header">Test Your Knowledge</div>', unsafe_allow_html=True)

        st.markdown("""
        Test your understanding of stationarity concepts with these practice examples.
        """)

        # Set up multiple questions
        questions = st.select_slider(
            "Select Question:",
            options=["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]
        )

        if questions == "Question 1":
            # Generate series
            np.random.seed(1)
            series = np.cumsum(np.random.normal(0, 1, 300))

            fig = plot_time_series(series, "Question 1: Analyze this series")
            st.plotly_chart(fig, use_container_width=True)

            q1_options = [
                "This series is stationary",
                "This series is non-stationary with a unit root",
                "This series is trend stationary",
                "This series is seasonally non-stationary"
            ]

            q1_answer = st.radio("What is the correct characterization?", q1_options)

            show_answer = st.button("Show Answer")

            if show_answer:
                st.markdown(
                    '<div class="info-box">Correct answer: This series is non-stationary with a unit root. This is a random walk (integrated of order 1), which has a stochastic trend and no tendency to return to a fixed mean.</div>',
                    unsafe_allow_html=True)

                # Show transformation
                diff_series = np.diff(series)
                fig_diff = plot_time_series(diff_series, "First Difference (Stationary)")
                st.plotly_chart(fig_diff, use_container_width=True)

                adf_output, adf_conclusion = adf_test_report(diff_series)
                st.write(f"**ADF p-value of differenced series:** {adf_output['p-value']:.4f} - {adf_conclusion}")

        elif questions == "Question 2":
            # Generate AR(1) process
            np.random.seed(2)
            ar_params = np.array([1, -0.7])
            series = arma_generate_sample(ar=ar_params, ma=[1], nsample=300)

            fig = plot_time_series(series, "Question 2: Analyze this series")
            st.plotly_chart(fig, use_container_width=True)

            # ACF plot
            acf_values = sm.tsa.acf(series, nlags=20)
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Scatter(
                x=list(range(len(acf_values))),
                y=acf_values,
                mode='markers+lines',
                name='ACF',
                line=dict(color='blue')
            ))

            # Add confidence intervals
            conf_level = 1.96 / np.sqrt(len(series))
            fig_acf.add_trace(go.Scatter(
                x=list(range(len(acf_values))),
                y=[conf_level] * len(acf_values),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Upper CI'
            ))
            fig_acf.add_trace(go.Scatter(
                x=list(range(len(acf_values))),
                y=[-conf_level] * len(acf_values),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Lower CI'
            ))

            fig_acf.update_layout(
                title="ACF of Series",
                xaxis_title='Lag',
                yaxis_title='Correlation',
                height=300
            )
            st.plotly_chart(fig_acf, use_container_width=True)

            q2_options = [
                "This series is non-stationary",
                "This series is stationary",
                "This is a random walk process",
                "This series is stationary after first differencing"
            ]

            q2_answer = st.radio("What is the correct characterization?", q2_options)

            show_answer = st.button("Show Answer")

            if show_answer:
                st.markdown(
                    '<div class="info-box">Correct answer: This series is stationary. This is an AR(1) process with φ=0.7, which is stationary because |φ| < 1. The ACF showing gradual decay is characteristic of stationary AR processes.</div>',
                    unsafe_allow_html=True)

                adf_output, adf_conclusion = adf_test_report(series)
                st.write(f"**ADF p-value:** {adf_output['p-value']:.4f} - {adf_conclusion}")

        elif questions == "Question 3":
            # Generate trend stationary series
            np.random.seed(3)
            t = np.arange(300)
            trend = 0.05 * t
            stationary_component = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=300)
            series = trend + stationary_component

            fig = plot_time_series(series, "Question 3: Analyze this series")
            st.plotly_chart(fig, use_container_width=True)

            q3_options = [
                "This series has a unit root",
                "This series is trend stationary",
                "This series is strictly stationary",
                "This series is a random walk with drift"
            ]

            q3_answer = st.radio("What is the correct characterization?", q3_options)

            show_answer = st.button("Show Answer")

            if show_answer:
                st.markdown(
                    '<div class="info-box">Correct answer: This series is trend stationary. It has a deterministic linear trend, and the deviations from this trend are stationary. This can be made stationary by either detrending or differencing.</div>',
                    unsafe_allow_html=True)

                # Show detrended series
                model = np.polyfit(t, series, 1)
                trend_fit = model[0] * t + model[1]
                detrended = series - trend_fit

                fig_detrend = plot_time_series(detrended, "Detrended Series (Stationary)")
                st.plotly_chart(fig_detrend, use_container_width=True)

                adf_output, adf_conclusion = adf_test_report(detrended)
                st.write(f"**ADF p-value of detrended series:** {adf_output['p-value']:.4f} - {adf_conclusion}")

        elif questions == "Question 4":
            # Generate series with structural break
            np.random.seed(4)
            n_samples = 300
            break_point = 150

            base_series = arma_generate_sample(ar=[1, -0.7], ma=[1], nsample=n_samples)
            series = add_structural_break(base_series, break_point, 'mean', mean_change=3.0)

            fig = plot_time_series(series, "Question 4: Analyze this series")
            st.plotly_chart(fig, use_container_width=True)

            # Run ADF test
            result = adfuller(series)

            st.write(f"**ADF Statistic:** {result[0]:.4f}")
            st.write(f"**p-value:** {result[1]:.4f}")

            q4_options = [
                "This series is truly non-stationary with a unit root",
                "This series is non-stationary due to a structural break",
                "This series is trend stationary",
                "This series is stationary around zero mean"
            ]

            q4_answer = st.radio("What is the correct characterization?", q4_options)

            show_answer = st.button("Show Answer")

            if show_answer:
                st.markdown(
                    '<div class="info-box">Correct answer: This series is non-stationary due to a structural break. The underlying process is stationary, but a mean shift at the break point makes it appear non-stationary to standard tests. This illustrates why structural breaks can reduce the power of unit root tests.</div>',
                    unsafe_allow_html=True)

                # Test series before and after break
                before_break = series[:break_point]
                after_break = series[break_point:]

                adf_before, _ = adf_test_report(before_break)
                adf_after, _ = adf_test_report(after_break)

                st.write(f"**ADF p-value before break:** {adf_before['p-value']:.4f}")
                st.write(f"**ADF p-value after break:** {adf_after['p-value']:.4f}")

                st.markdown(
                    "Notice how the series is more likely to be identified as stationary when tested in the separate regimes (before/after break).")

        elif questions == "Question 5":
            # Generate near unit root process
            np.random.seed(5)
            ar_params = np.array([1, -0.95])
            series = arma_generate_sample(ar=ar_params, ma=[1], nsample=100)  # Small sample

            fig = plot_time_series(series, "Question 5: Analyze this series (n=100)")
            st.plotly_chart(fig, use_container_width=True)

            # Run ADF test
            result = adfuller(series)

            st.write(f"**ADF Statistic:** {result[0]:.4f}")
            st.write(f"**p-value:** {result[1]:.4f}")

            q5_options = [
                "This series has a unit root",
                "This series is stationary but the test lacks power to detect it",
                "This series is trend stationary",
                "This series has an explosive root"
            ]

            q5_answer = st.radio("What is the correct characterization?", q5_options)

            show_answer = st.button("Show Answer")

            if show_answer:
                st.markdown(
                    '<div class="info-box">Correct answer: This series is stationary but the test lacks power to detect it. This is an AR(1) process with φ=0.95, which is technically stationary (|φ| < 1), but very close to the unit root boundary. With a small sample (n=100), the ADF test often lacks power to distinguish this from a true unit root process.</div>',
                    unsafe_allow_html=True)

                # Generate longer series for comparison
                long_series = arma_generate_sample(ar=ar_params, ma=[1], nsample=1000)

                # Run ADF test on longer series
                long_result = adfuller(long_series)

                st.write(f"**ADF p-value with larger sample (n=1000):** {long_result[1]:.4f}")

                st.markdown(
                    "With a larger sample, the test has more power to correctly identify the series as stationary.")

st.sidebar.markdown("""
### Key Takeaways

1. **Stationarity** is crucial for valid time series analysis
2. **Unit root tests** help determine if a series is stationary
3. **Structural breaks** can reduce test power and lead to incorrect conclusions
4. **Multiple testing approaches** provide more robust inference
5. **Appropriate transformations** can convert non-stationary series to stationary ones
""")

st.sidebar.markdown("""
### About This App

This educational app was designed to help teachers and students understand the concepts of stationarity and unit root testing in time series analysis. It provides interactive visualizations, examples, and exercises to reinforce learning.
""")

