import streamlit as st
import plotly.graph_objects as go
from backend import MarketEngine
import pandas as pd

# Page Config
st.set_page_config(page_title="Financial Vision-India: Live", layout="wide", page_icon="📡")

# Custom CSS
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize Engine
# @st.cache_resource # Disabled for debugging/development
def load_engine():
    return MarketEngine()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load Market Engine: {e}")
    st.stop()

# --- Sidebar: Market Watch ---
st.sidebar.title("📡 NSE Market Watch (v2.1)")

# Mode Selector - Default to Backtest Lab as requested
app_mode = st.sidebar.selectbox("Select Mode", ["Backtest Lab", "Live Scanner"])

# Custom CSS for High Contrast Metric Cards
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    padding: 10px;
    border-radius: 5px;
    color: white;
}
div[data-testid="stMetricLabel"] {
    color: #AAAAAA !important;
}
div[data-testid="stMetricValue"] {
    color: #00FF00 !important;
}
</style>
""", unsafe_allow_html=True)

if app_mode == "Live Scanner":
    st.sidebar.subheader("Live Feed")
    
    search_term = st.sidebar.text_input("Search Ticker", "")
    tickers = engine.get_nifty_100_tickers()
    
    filtered_tickers = [t for t in tickers if search_term.upper() in t]
    
    selected_ticker = st.sidebar.radio("Select Stock", filtered_tickers)
    
    if st.sidebar.button("Force Refresh Data"):
        st.cache_data.clear()
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Tracking {len(tickers)} Assets")
    
    # --- Main Panel: Live Scanner ---
    
    if selected_ticker:
        st.title(f"Analyzing: {selected_ticker}")
        
        with st.spinner("Fetching Live Market Data & Running Vision Model..."):
            # Run Analysis
            status = engine.run_live_analysis(selected_ticker)
        
        if status:
            # Check Source
            if status.get('data_source') and status['data_source'] != 'LIVE':
                st.warning(f"⚠️ **{status['data_source']}** - Displaying synthetic data for prototype demonstration.")
                
            # Header
            st.markdown(f"### Current Price: ₹ {status['current_price']:.2f}")
            
            # Row 1: Visuals
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Price History (50 Hrs)")
                df = status['recent_prices']
                # df columns might include Ticker name if MultiIndex not perfectly flattened, 
                # or just Date Open High Low Close
                # 'Date' is likely index if we reset_index it is column
                
                # Identify Date col
                x_col = 'Date' if 'Date' in df.columns else df.index.name or 'index'
                # y finance creates 'Datetime' or 'Date'
                
                # Fix for "Disconnected Graph" (Gaps): Use Category Axis
                # Convert timestamps to strings for category plotting
                df['DateStr'] = df.iloc[:, 0].dt.strftime('%Y-%m-%d %H:%M')
                
                fig = go.Figure(data=[go.Candlestick(x=df['DateStr'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])])
                
                # Remove range slider and force category type
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
                fig.update_xaxes(type='category', nticks=10) # Show ~10 labels to avoid clutter
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("The GAF Vision Input")
                st.image(status['vision_image'], caption="AI Input (64x64 Grayscale)", width=200, clamp=True, output_format='PNG')
                
            # Row 2: Gatekeeper Cards
            st.markdown("---")
            st.subheader("🛡️ Gatekeeper Diagnostics")
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric("Volatility Status", status['volatility_status'], delta=status['volatility_val'], delta_color="inverse")
                
            with c2:
                st.metric("Pattern Detected", status['pattern_detected'])
                
            with c3:
                dec = status['ai_decision']
                if dec == "BUY":
                    st.success(f"AI VERDICT: {dec}")
                elif dec == "BLOCKED":
                    st.error(f"AI VERDICT: {dec}")
                else:
                    st.warning(f"AI VERDICT: {dec}")
    
            # Row 3: Explainability
            with st.expander("📝 Decision Logic Log (Live)", expanded=True):
                for i, line in enumerate(status['gatekeeper_log']):
                    st.text(f"{i+1}. {line}")
                    
        else:
            st.error("Failed to fetch data or data insufficient for analysis.")

elif app_mode == "Backtest Lab":
    st.sidebar.header("🧪 Experiment Setup")
    
    bt_ticker = st.sidebar.selectbox("Select Asset", engine.get_nifty_100_tickers(), index=0)
    bt_days = st.sidebar.slider("History Duration (Days)", 30, 90, 60)
    
    st.title("🧪 PPO Backtest Lab")
    st.info(f"Simulating PPO Agent Strategy on {bt_ticker} for last {bt_days} days...")
    
    if st.button("RUN PPO AGENT SIMULATION", type="primary"):
        with st.spinner("Running PPO Inference Loop..."):
            res = engine.run_backtest(bt_ticker, days=bt_days)
            
        if "error" in res:
             st.error(f"Backtest Failed: {res['error']}")
        else:
             # Metrics with styling via CSS above
             m1, m2, m3 = st.columns(3)
             m1.metric("Final ROI", f"{res['roi']:.2f}%")
             m2.metric("Total Trades", res['total_trades'])
             m3.metric("Final Equity", f"₹ {res['final_equity']:.2f}")
             
             # Chart with Neon Green
             st.subheader("Equity Curve (Virtual Portfolio)")
             edf = res['equity_df']
             fig = go.Figure()
             # Use Neon Green (#00FF00) for high visibility
             fig.add_trace(go.Scatter(x=edf['Date'], y=edf['Equity'], mode='lines', name='PPO Agent Equity', line=dict(color='#00FF00', width=3)))
             fig.update_layout(height=450, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
             st.plotly_chart(fig, use_container_width=True)
             
             # Trades
             st.subheader("Trade Log")
             if res['trades']:
                 tdf = pd.DataFrame(res['trades'])
                 st.dataframe(tdf, use_container_width=True)
             else:
                 st.warning("No trades executed by the AI (Gatekeeper likely blocked all entries).")
