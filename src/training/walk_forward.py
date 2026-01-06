import pandas as pd
import numpy as np
from datetime import timedelta
from src.training.backtester import Backtester
from src.training.trading_env import TradingEnv
from src.features.indicators import IndicatorEngine
from src.features.feature_builder import FeatureBuilder

class WalkForwardValidator:
    def __init__(self, train_days=20, test_days=5, step_days=5):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def run(self, agent, df):
        """
        Executes walk-forward validation on the given dataframe.
        """
        results = []
        
        # Ensure we have enough data
        if df.empty:
            return []

        # Assuming the index is datetime or we have a Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            start_date = df['Date'].min()
            end_date = df['Date'].max()
        else:
            print("Error: DataFrame must have a 'Date' column for Walk-Forward Validation.")
            return []

        current_train_start = start_date
        
        while True:
            train_end = current_train_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)
            
            if test_end > end_date:
                break
                
            # Filter data for this fold
            train_data = df[(df['Date'] >= current_train_start) & (df['Date'] < train_end)]
            test_data = df[(df['Date'] >= train_end) & (df['Date'] < test_end)]
            
            if len(train_data) < 100 or len(test_data) < 20:
                current_train_start += timedelta(days=self.step_days)
                continue

            print(f"Validating Fold: Train {train_data['Date'].min()} to {train_data['Date'].max()} | Test {test_data['Date'].min()} to {test_data['Date'].max()}")
            
            # Setup features for this fold
            ind_engine = IndicatorEngine()
            train_df = ind_engine.compute_all(train_data)
            test_df = ind_engine.compute_all(test_data)
            
            fb = FeatureBuilder()
            train_features = fb.build_features(train_df)
            test_features = fb.build_features(test_df)
            
            # Run Backtest on the test window
            env = TradingEnv(test_data, test_features, asset_name="WF-Fold")
            bt = Backtester(agent, env)
            result = bt.run()
            
            results.append({
                "fold_start": train_end,
                "fold_end": test_end,
                "metrics": result
            })
            
            current_train_start += timedelta(days=self.step_days)
            
        return results
