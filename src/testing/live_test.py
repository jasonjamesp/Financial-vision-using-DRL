import time
import json
import torch
import numpy as np
from pathlib import Path
from src.features.indicators import IndicatorEngine
from src.features.feature_builder import FeatureBuilder

class LiveTester:
    def __init__(self, config_path="config/trading_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = TradingLogger()
        self.dm = DataManager()
        self.gaf_encoder = GAFEncoder()
        self.feature_builder = FeatureBuilder()
        
        self.agent = PPOAgent()
        self._load_model()
        
        self.balance = self.config.get("initial_balance", 10000.0)
        self.shares_held = 0.0
        self.max_net_worth = self.balance

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _load_model(self):
        checkpoint = self.config.get("model_checkpoint")
        if checkpoint and Path(checkpoint).exists():
            print(f"Loading model from {checkpoint}")
            self.agent.load(checkpoint)
        else:
            print("Warning: No model checkpoint found. Using random initialized agent.")

    def run(self):
        print("Starting Live Testing Mode (Enhanced ML)...")
        asset = self.config.get("assets", ["ETH-USD"])[0]
        
        while True:
            # Hot-reload config
            try:
                self.config = self._load_config()
            except Exception as e:
                print(f"Error reloading config: {e}")

            # 1. Fetch Live Data (Last 100 candles to compute indicators)
            df = self.dm.fetch_ohlcv(asset, days=5) 
            if df.empty or len(df) < 64:
                print("Insufficient data. Retrying in 60s...")
                time.sleep(60)
                continue
            
            # 2. Compute Indicators and Features
            df = IndicatorEngine.compute_all(df)
            feature_df = self.feature_builder.build_features(df)
            
            # Last available step
            latest_idx = -1
            window = df['close'].iloc[-64:].values
            gaf_image = self.gaf_encoder.encode(window)
            indicators = feature_df.iloc[latest_idx].values.astype(np.float32)
            
            current_price = window[-1]
            net_worth = self.balance + (self.shares_held * current_price)
            self.max_net_worth = max(self.max_net_worth, net_worth)
            
            drawdown = (self.max_net_worth - net_worth) / (self.max_net_worth + 1e-8)
            pnl_pct = (net_worth - self.config.get("initial_balance", 10000.0)) / (self.config.get("initial_balance", 10000.0) + 1e-8)
            
            state = np.array([
                self.balance,
                self.shares_held,
                net_worth,
                drawdown,
                pnl_pct
            ], dtype=np.float32)
            
            # 3. Get Decision from Agent
            action, log_prob = self.agent.select_action(
                gaf_image.reshape(1, 64, 64), 
                indicators,
                state
            )
            
            # 4. Execute Paper Trade
            action_desc = "HOLD"
            if action == 1: # Buy All
                if self.balance > 0:
                    shares_to_buy = (self.balance * self.config.get("position_size_pct", 1.0)) / current_price
                    self.shares_held += shares_to_buy
                    self.balance -= shares_to_buy * current_price
                    action_desc = f"BUY {shares_to_buy:.4f}"
            elif action == 2: # Sell All
                if self.shares_held > 0:
                    self.balance += self.shares_held * current_price
                    action_desc = f"SELL {self.shares_held:.4f}"
                    self.shares_held = 0
            
            # 5. Log Decision
            self.logger.log_trade(
                asset=asset,
                action=action_desc,
                price=current_price,
                balance=self.balance,
                shares=self.shares_held,
                net_worth=net_worth,
                confidence=float(torch.exp(log_prob))
            )
            
            print(f"[{time.strftime('%H:%M:%S')}] Asset: {asset} | Action: {action_desc} | Net Worth: {net_worth:.2f}")
            time.sleep(self.config.get("poll_interval_seconds", 60))

if __name__ == "__main__":
    tester = LiveTester()
    tester.run()
