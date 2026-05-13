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
def fetch_and_process_data(ticker, days=60, target_tz=None):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    # Fetch 5-minute interval data
    df = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False)
    
    if df.empty:
        return None

    # yfinance may return MultiIndex columns (e.g., ('Close', 'SPY')); flatten to base names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.reset_index(inplace=True)
    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'

    if time_col not in df.columns:
        return None

    # Ensure datetime is timezone-aware (yfinance often returns UTC). Convert to target_tz if requested.
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    if target_tz:
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
    days = st.slider("Lookback Period (Days)", min_value=10, max_value=60, value=60, step=10, help="Yahoo Finance limits 5m data to 60 days max.")
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
with st.spinner(f"Fetching {days} days of 5-minute data for {ticker}..."):
    raw_df = fetch_and_process_data(ticker, days, target_tz=target_tz)

if raw_df is None or raw_df.empty:
    st.error(f"Failed to fetch data for {ticker}. It might be delisted, or Yahoo Finance is blocking the request.")
else:
    # Run Analysis
    pattern_df = analyze_patterns(raw_df)

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
    ny_reference_date = raw_df['Date_Only'].max() if 'Date_Only' in raw_df.columns else datetime.datetime.now().date()
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
    st.markdown("This chart shows the average percentage deviation from the daily opening price at every 5-minute interval.")
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