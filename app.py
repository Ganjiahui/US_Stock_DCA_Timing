import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np


def convert_time_str_between_timezones(time_str, from_tz, to_tz, base_date=None):
    """Convert an HH:MM time string between timezones with DST-aware rules."""
    if base_date is None:
        base_date = datetime.datetime.now().date()

    naive_dt = datetime.datetime.strptime(f"{base_date} {time_str}", "%Y-%m-%d %H:%M")

    try:
        from zoneinfo import ZoneInfo
        source_dt = naive_dt.replace(tzinfo=ZoneInfo(from_tz))
        target_dt = source_dt.astimezone(ZoneInfo(to_tz))
    except Exception:
        try:
            import pytz
            source_tz = pytz.timezone(from_tz)
            target_tz = pytz.timezone(to_tz)
            source_dt = source_tz.localize(naive_dt)
            target_dt = source_dt.astimezone(target_tz)
        except Exception:
            return None

    return target_dt.strftime('%H:%M')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Intraday DCA Optimizer", page_icon="📈", layout="wide")

# --- DATA PROCESSING FUNCTIONS ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent API spam
def fetch_and_process_data(ticker, days=60, target_tz=None, interval="5m"):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)

    # Fetch interval-based data. 5m is for intraday mode, 1d is for multi-year trend mode.
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    
    if df.empty:
        return None

    # yfinance may return MultiIndex columns (e.g., ('Close', 'SPY')); flatten to base names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.reset_index(inplace=True)
    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'

    if time_col not in df.columns:
        return None

    # Keep intraday timestamps timezone-aware; daily candles can remain date-like.
    force_utc = interval != "1d"
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=force_utc)
    if target_tz and interval != "1d":
        try:
            df[time_col] = df[time_col].dt.tz_convert(target_tz)
        except Exception:
            # If conversion fails, try localize then convert
            df[time_col] = df[time_col].dt.tz_localize('UTC').dt.tz_convert(target_tz)

    # Feature Engineering
    df['Time'] = df[time_col].dt.time
    df['Date_Only'] = df[time_col].dt.date
    
    # Ensure we only use full trading days (simplification for market hours)
    # Calculate % change from the daily open to find intraday structure
    df['Daily_Open'] = df.groupby('Date_Only')['Open'].transform('first')
    df['Dev_From_Open_Pct'] = ((df['Close'] - df['Daily_Open']) / df['Daily_Open']) * 100
    
    return df

def analyze_patterns(df):
    # Group by Time to find the average deviation from the open at each specific time bucket
    time_stats = df.groupby('Time').agg(
        Avg_Dev_Pct=('Dev_From_Open_Pct', 'mean'),
        Median_Dev_Pct=('Dev_From_Open_Pct', 'median'),
        Volatility=('Dev_From_Open_Pct', 'std'),
        Count=('Dev_From_Open_Pct', 'count')
    ).reset_index()
    
    # Filter out weird after-hours artifacts by ensuring enough data points
    max_count = time_stats['Count'].max()
    time_stats = time_stats[time_stats['Count'] >= (max_count * 0.8)]
    
    return time_stats

# --- USER INTERFACE ---
st.title("⚡ Intraday DCA Timing Optimizer")
st.markdown("Identify the statistically optimal time of day to execute your Dollar-Cost Averaging buys based on historical intraday price patterns.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Strategy Parameters")
    ticker = st.text_input("Asset Ticker (e.g., SPY, AAPL, QQQ)", value="SPY").upper()
    lookback_options = {
        "2 y": 365 * 2,
        "1 y": 365,
        "Half a year": 182,
        "Quarter of year": 91,
        "60 days": 60,
        "30 days": 30,
    }
    lookback_label = st.selectbox(
        "History Date Range",
        options=list(lookback_options.keys()),
        index=0,
        help="Intraday timing metrics use 5-minute data (latest 60 days max). Longer ranges also show a separate daily trend section."
    )
    days = lookback_options[lookback_label]
    dca_amount = st.number_input("DCA Order Size ($)", min_value=10, max_value=10000, value=1000, step=100)
    fee_pct = st.number_input("Brokerage Fee (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
    st.divider()
    # Time display option: allow user to view Malaysia time (UTC+8)
    tz_option = st.selectbox("Display Timezone", options=["Original (UTC)", "Malaysia (UTC+8)"], index=0)
    target_tz = 'Asia/Kuala_Lumpur' if tz_option == 'Malaysia (UTC+8)' else None

if not ticker:
    st.warning("Please enter a valid ticker symbol.")
    st.stop()

# --- EXECUTION LOGIC ---
MAX_5M_DAYS = 60
MAX_1H_DAYS = 730

# Intraday timing window selection based on Yahoo interval limits.
if days <= MAX_5M_DAYS:
    intraday_interval = "5m"
    intraday_label = "5-minute"
    intraday_days = days
else:
    intraday_interval = "1h"
    intraday_label = "1-hour"
    intraday_days = min(days, MAX_1H_DAYS)

show_daily_trend = days > MAX_5M_DAYS

if days > MAX_5M_DAYS and days <= MAX_1H_DAYS:
    st.info(
        f"For {lookback_label}, timing metrics are computed using {intraday_label} data across the selected period. "
        "(Yahoo limits 5-minute data to 60 days.)"
    )
elif days > MAX_1H_DAYS:
    st.warning(
        f"Yahoo intraday history is limited to ~{MAX_1H_DAYS} days for 1-hour data. "
        f"Timing metrics for {lookback_label} are computed from the latest {MAX_1H_DAYS} days."
    )

with st.spinner(f"Fetching last {intraday_days} days of {intraday_label} intraday data for {ticker}..."):
    intraday_df = fetch_and_process_data(ticker, intraday_days, target_tz=target_tz, interval=intraday_interval)

daily_df_full = None
if show_daily_trend:
    with st.spinner(f"Fetching {lookback_label.lower()} of daily (1d) trend data for {ticker}..."):
        daily_df_full = fetch_and_process_data(ticker, days, target_tz=target_tz, interval="1d")

if intraday_df is None or intraday_df.empty:
    st.error(f"Failed to fetch data for {ticker}. It might be delisted, or Yahoo Finance is blocking the request.")
else:
    # --- INTRADAY TIMING ANALYSIS (always shown) ---
    pattern_df = analyze_patterns(intraday_df)

    # Sort chronologically by Time so x-axis ordering matches human expectation
    pattern_df = pattern_df.sort_values(by='Time').reset_index(drop=True)

    # Create consistent time string used for plotting and annotation so x-values match exactly
    pattern_df['Time_Str'] = pattern_df['Time'].apply(lambda t: t.strftime('%H:%M') if hasattr(t, 'strftime') else str(t))

    # Find Optimal Times (recompute after sorting to avoid index confusion)
    best_time_row = pattern_df.loc[pattern_df['Avg_Dev_Pct'].idxmin()]
    worst_time_row = pattern_df.loc[pattern_df['Avg_Dev_Pct'].idxmax()]

    best_time_str = best_time_row['Time'].strftime('%H:%M') if hasattr(best_time_row['Time'], 'strftime') else str(best_time_row['Time'])
    worst_time_str = worst_time_row['Time'].strftime('%H:%M') if hasattr(worst_time_row['Time'], 'strftime') else str(worst_time_row['Time'])
    edge_pct = worst_time_row['Avg_Dev_Pct'] - best_time_row['Avg_Dev_Pct']

    # Use the latest analyzed market date so DST conversion reflects the dataset period.
    ny_reference_date = intraday_df['Date_Only'].max() if 'Date_Only' in intraday_df.columns else datetime.datetime.now().date()
    best_time_my_from_ny = convert_time_str_between_timezones(
        best_time_str,
        from_tz='US/Eastern',
        to_tz='Asia/Kuala_Lumpur',
        base_date=ny_reference_date
    )
    worst_time_my_from_ny = convert_time_str_between_timezones(
        worst_time_str,
        from_tz='US/Eastern',
        to_tz='Asia/Kuala_Lumpur',
        base_date=ny_reference_date
    )
    
    # Calculate simulated metrics
    fee_cost = dca_amount * (fee_pct / 100)
    savings_per_order = dca_amount * (edge_pct / 100)

    # --- DASHBOARD RENDERING ---
    st.divider()
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal Buy Time", best_time_str, f"{best_time_row['Avg_Dev_Pct']:.3f}% avg dip", delta_color="inverse")
    col2.metric("Worst Buy Time", worst_time_str, f"{worst_time_row['Avg_Dev_Pct']:.3f}% avg premium")
    col3.metric("Timing Edge (%)", f"{edge_pct:.3f}%", "Difference between Best & Worst")
    col4.metric("Est. Edge per Order ($)", f"${savings_per_order:.2f}", f"Minus ${fee_cost:.2f} fees")

    st.divider()
    
    # Main Chart
    st.subheader(f"📊 Intraday Price Behavior for {ticker}")
    st.markdown(
        f"This chart shows the average percentage deviation from the daily opening price at each {intraday_label} bucket."
    )
    st.caption(f"Times shown in: {tz_option}")
    if best_time_my_from_ny and worst_time_my_from_ny:
        st.caption(
            f"NY (US/Eastern) -> Malaysia (DST-aware): Best {best_time_str} -> {best_time_my_from_ny}, "
            f"Worst {worst_time_str} -> {worst_time_my_from_ny}"
        )
    
    # Plotly Line Chart
    fig = px.line(
        pattern_df,
        x='Time_Str',
        y='Avg_Dev_Pct',
        markers=True,
        labels={'x': 'Time of Day', 'Avg_Dev_Pct': 'Avg Deviation from Daily Open (%)'},
        template="plotly_white"
    )
    
    # Highlight Best Time
    fig.add_annotation(
        x=best_time_str,
        y=best_time_row['Avg_Dev_Pct'],
        text="Target Entry",
        showarrow=True,
        arrowhead=2,
        arrowcolor="green",
        font=dict(color="green", size=14)
    )
    
    # Highlight Worst Time
    fig.add_annotation(
        x=worst_time_str,
        y=worst_time_row['Avg_Dev_Pct'],
        text="Avoid Entry",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red",
        font=dict(color="red", size=14)
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Data Table Expander
    with st.expander("Show Raw Data Table"):
        st.dataframe(pattern_df.sort_values(by='Avg_Dev_Pct'), width='stretch')

if show_daily_trend and daily_df_full is not None and not daily_df_full.empty:
    # --- DAILY TREND ANALYSIS (1D candles for >60 day lookbacks) ---
    daily_df = daily_df_full.sort_values(by='Date_Only').drop_duplicates(subset='Date_Only', keep='last').copy()
    daily_df['Daily_Return_Pct'] = daily_df['Close'].pct_change() * 100
    daily_df['SMA_50'] = daily_df['Close'].rolling(window=50).mean()
    daily_df['SMA_200'] = daily_df['Close'].rolling(window=200).mean()

    first_close = float(daily_df['Close'].iloc[0])
    last_close = float(daily_df['Close'].iloc[-1])
    total_return_pct = ((last_close / first_close) - 1) * 100 if first_close != 0 else 0.0

    days_span = max((daily_df['Date_Only'].iloc[-1] - daily_df['Date_Only'].iloc[0]).days, 1)
    years_span = days_span / 365.25
    cagr_pct = (((last_close / first_close) ** (1 / years_span)) - 1) * 100 if years_span > 0 and first_close > 0 else np.nan
    annual_vol_pct = daily_df['Daily_Return_Pct'].dropna().std() * np.sqrt(252)

    st.divider()
    st.subheader(f"📈 Multi-Year Trend Analysis for {ticker}")
    st.markdown("Long lookbacks use daily candles to show trend direction and volatility.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close ($)", f"${last_close:.2f}", f"From ${first_close:.2f}")
    col2.metric("Total Return (%)", f"{total_return_pct:.2f}%", f"Over ~{years_span:.2f} years")
    col3.metric("CAGR (%)", f"{cagr_pct:.2f}%" if np.isfinite(cagr_pct) else "N/A")
    col4.metric("Annualized Volatility (%)", f"{annual_vol_pct:.2f}%" if np.isfinite(annual_vol_pct) else "N/A")

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=daily_df['Date_Only'],
        y=daily_df['Close'],
        mode='lines',
        name='Close',
        line=dict(color='#1f77b4', width=2)
    ))
    trend_fig.add_trace(go.Scatter(
        x=daily_df['Date_Only'],
        y=daily_df['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(color='#ff7f0e', width=1.5)
    ))
    trend_fig.add_trace(go.Scatter(
        x=daily_df['Date_Only'],
        y=daily_df['SMA_200'],
        mode='lines',
        name='SMA 200',
        line=dict(color='#2ca02c', width=1.5)
    ))
    trend_fig.update_layout(
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend_title='Series'
    )

    st.plotly_chart(trend_fig, width='stretch')

    with st.expander("Show Daily Data Table"):
        st.dataframe(
            daily_df[['Date_Only', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return_Pct']].sort_values(by='Date_Only', ascending=False),
            width='stretch'
        )