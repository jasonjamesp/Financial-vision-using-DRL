import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
import yfinance as yf
import mplfinance as mpf
import cv2
import io
import matplotlib.pyplot as plt

import requests_cache

# Matplotlib backend for non-GUI
plt.switch_backend('Agg')

class MarketEngine:
    def __init__(self, model_path='./models/eth_gatekeeper_final.zip'):
        self.model_path = model_path
        self.model = None
        
        # Initialize Cache Session (1 Hour expiry)
        self.session = requests_cache.CachedSession('yfinance_cache', expire_after=3600)
        
        # Load Model
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Warning: Model not found at {self.model_path}")

    def get_nifty_100_tickers(self):
        """Returns a list of top NSE tickers."""
        # Top 20 for prototype speed
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
            "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS",
            "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "TATASTEEL.NS",
            "ETH-USD" # Crypto Benchmark
        ]

    def run_live_analysis(self, ticker):
        """
        Fetches live data and runs the full pipeline for a single ticker.
        Includes Fallback to Synthetic Data if API fails.
        """
        print(f"--- Analyzing {ticker} ---")
        df = pd.DataFrame()
        data_source = "LIVE"
        
        try:
            # 1. Fetch Data
            # Use Ticker object (Modern yfinance)
            # requests_cache session disabled due to conflict with curl_cffi
            dat = yf.Ticker(ticker)
            df = dat.history(period='60d', interval='1h')
            
            # Validation
            if df.empty or len(df) < 50:
                # Try without cache if empty? Or just fail.
                # If cached result was empty, maybe force refresh? 
                # For now, strict fail.
                raise ValueError("Insufficient Data (Empty)")
                
            # Formatting
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Ensure proper columns
            df.columns = [c.capitalize() for c in df.columns]
            
        except Exception as e:
            print(f"Live Data Failed for {ticker}: {e}. Switching to SIMULATION.")
            df = self._generate_synthetic_data(ticker)
            data_source = "SIMULATED (Live Feed Internal Error)"

        try:
            # Get last window
            window = df.iloc[-50:].copy() 
            current_price = window['Close'].iloc[-1]
            
            # 2. Process Image (Visual Vision)
            buf = io.BytesIO()
            mc = mpf.make_marketcolors(up='white', down='black', edge='black', wick='black', volume='in')
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', facecolor='white')
            
            mpf.plot(window, type='candle', style=s, savefig=dict(fname=buf, dpi=100, bbox_inches='tight'), axisoff=True, volume=False)
            buf.seek(0)
            
            file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            
            # 3. Features
            window['Returns'] = window['Close'].pct_change()
            vol = window['Returns'].std() * np.sqrt(252*7)
            vol_val = window['Returns'].std()
            vol_flag = 1 if vol_val < 0.005 else 0 
            
            # Patterns (Bullish Engulfing)
            p_open = window['Open'].iloc[-2]
            p_close = window['Close'].iloc[-2]
            c_open = window['Open'].iloc[-1]
            c_close = window['Close'].iloc[-1]
            
            raw_pattern = 0 
            pat_name = "None"
            
            if (p_close < p_open) and (c_close > c_open):
                if (c_open < p_close) and (c_close > p_open):
                    raw_pattern = 1
                    pat_name = "Bullish Engulfing"
            
            # 4. Gatekeeper Logic
            log = []
            decision = "HOLD"
            status_color = "green"
            
            if data_source != "LIVE":
                log.append(f"⚠️ WARNING: DATA SOURCE IS {data_source}")
            
            if vol_flag == 1:
                log.append(f"Step 1: Volatility Check -> PASSED (StdDev: {vol_val:.4f})")
            else:
                log.append(f"Step 1: Volatility Check -> FAILED (High Risk: {vol_val:.4f})")
                
            gatekeeper_pass = (vol_flag == 1)
            
            if gatekeeper_pass:
                if raw_pattern != 0:
                    log.append(f"Step 2: Pattern Check -> PASSED ({pat_name})")
                else:
                    log.append("Step 2: Pattern Check -> FAILED (No Pattern)")
                    gatekeeper_pass = False 
                    
            if gatekeeper_pass and self.model:
                 obs_img = np.expand_dims(img_resized, axis=-1)
                 obs = {
                    "image": np.expand_dims(obs_img, axis=0),
                    "chips": np.array([0]),
                    "pattern": np.array([raw_pattern + 1])
                 }
                 action, _ = self.model.predict(obs)
                 action_code = int(action[0])
                 mp = {0: "HOLD", 1: "BUY", 2: "SELL"}
                 decision = mp.get(action_code, "HOLD")
                 log.append(f"Step 3: AI Confidence -> DECISION: {decision}")
            elif not gatekeeper_pass:
                 decision = "BLOCKED"
                 status_color = "red"
                 log.append("Step 3: AI Brain -> SKIPPED (Gatekeeper Block)")
            
            return {
                "current_price": current_price,
                "volatility_status": "Safe" if vol_flag == 1 else "Risky",
                "volatility_val": f"{vol_val:.2%}",
                "pattern_detected": pat_name,
                "ai_decision": decision,
                "gatekeeper_log": log,
                "vision_image": img_resized,
                "recent_prices": window.reset_index(),
                "data_source": data_source
            }
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_backtest(self, ticker, days=90):
        """
        Runs a simulation on historical data for the ticker.
        Returns metrics and equity curve.
        """
        print(f"--- Running Backtest for {ticker} ---")
        try:
            # 1. Fetch History
            dat = yf.Ticker(ticker)
            df = dat.history(period=f'{days}d', interval='1h')
            
            if len(df) < 100:
                return {"error": "Insufficient history for backtest"}

            # Reset logic
            balance = 100000.0 # Initial INR
            holdings = 0
            equity_curve = []
            trades = []
            
            # Use a rolling window simulation
            # We need at least 50 candles for the first image
            window_size = 50
            
            # Pre-calculate features to speed up?
            # For strictness, we should loop. 
            # Optimization: Just loop indices
            
            for i in range(window_size, len(df)):
                # Slice Window
                window = df.iloc[i-window_size:i].copy()
                current_price = window['Close'].iloc[-1]
                current_time = window.index[-1]
                
                # --- FAST FEATURE GEN (Simplified for Speed) ---
                # Image
                # Skipping full MPF plot for speed in backtest loop if possible?
                # No, model needs image. MPF is slow (0.1s). 60 days * 7 hours = 420 iterations. ~40 seconds. Acceptable.
                
                buf = io.BytesIO()
                mc = mpf.make_marketcolors(up='white', down='black', edge='black', wick='black', volume='in')
                s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', facecolor='white')
                mpf.plot(window, type='candle', style=s, savefig=dict(fname=buf, dpi=50, bbox_inches='tight'), axisoff=True, volume=False)
                buf.seek(0)
                file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                
                # Gatekeeper Stats
                window['Returns'] = window['Close'].pct_change()
                vol_val = window['Returns'].std()
                vol_flag = 1 if vol_val < 0.005 else 0
                
                # Pattern
                p_open = window['Open'].iloc[-2]
                p_close = window['Close'].iloc[-2]
                c_open = window['Open'].iloc[-1]
                c_close = window['Close'].iloc[-1]
                raw_pattern = 0
                if (p_close < p_open) and (c_close > c_open):
                    if (c_open < p_close) and (c_close > p_open):
                        raw_pattern = 1
                
                # AI Decision
                action_code = 0 # HOLD
                
                # Logic: Pass Gatekeeper -> Predict
                if vol_flag == 1:
                     if self.model:
                         obs_img = np.expand_dims(img_resized, axis=-1)
                         obs = {
                            "image": np.expand_dims(obs_img, axis=0),
                            "chips": np.array([0]), # Mock chips
                            "pattern": np.array([raw_pattern + 1])
                         }
                         action, _ = self.model.predict(obs)
                         action_code = int(action[0])
                else:
                    action_code = 0 # Blocked
                
                # Execute Trade
                if action_code == 1: # BUY
                     cost = current_price
                     if balance >= cost:
                         balance -= cost
                         holdings += 1
                         trades.append({'date': current_time, 'type': 'BUY', 'price': cost})
                         
                elif action_code == 2: # SELL
                     if holdings > 0:
                         revenue = current_price
                         balance += revenue
                         holdings -= 1
                         trades.append({'date': current_time, 'type': 'SELL', 'price': revenue})
                
                # Record Equity
                total_equity = balance + (holdings * current_price)
                equity_curve.append({'Date': current_time, 'Equity': total_equity})
                
            # Results
            if not equity_curve:
                 return {"error": "No data processed"}
                 
            edf = pd.DataFrame(equity_curve)
            initial = 100000.0
            final = edf['Equity'].iloc[-1]
            roi = ((final - initial) / initial) * 100
            
            return {
                "equity_df": edf,
                "trades": trades,
                "roi": roi,
                "final_equity": final,
                "total_trades": len(trades)
            }
            
        except Exception as e:
            print(f"Backtest Failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
            
    def _generate_synthetic_data(self, ticker):
        """Generates realistic Hourly OHLC data for fallback."""
        num_hours = 200
        start_price = 2500.0 if "RELIANCE" in ticker else 1000.0
        
        # Random Walk
        returns = np.random.normal(0, 0.002, num_hours) # 0.2% hourly vol
        price_path = start_price * (1 + returns).cumprod()
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=num_hours, freq='h')
        
        df = pd.DataFrame({'Close': price_path}, index=dates)
        
        # Add OHLC noise
        noise = df['Close'] * 0.001
        df['Open'] = df['Close'].shift(1).fillna(start_price)
        df['High'] = df[['Open', 'Close']].max(axis=1) + (noise * np.random.rand())
        df['Low'] = df[['Open', 'Close']].min(axis=1) - (noise * np.random.rand())
        
        return df
