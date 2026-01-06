import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureBuilder:
    def __init__(self):
        self.scaler = StandardScaler()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a normalized feature set from technical indicators.
        Returns a dataframe with features suitable for RL input.
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Momentum Features
        features['rsi_norm'] = df['rsi'] / 100.0
        features['macd_hist_norm'] = self._normalize(df['macd_hist'])
        features['stoch_k_norm'] = df['stoch_k'] / 100.0
        
        # 2. Volatility Features
        features['bb_pos'] = df['bb_percent']
        features['atr_norm'] = self._normalize(df['atr'])
        
        # 3. Trend Features
        features['adx_norm'] = df['adx'] / 100.0
        features['price_ema9_ratio'] = df['close'] / df['ema_9'] - 1
        features['ema9_21_ratio'] = df['ema_9'] / df['ema_21'] - 1
        
        # 4. Volume Features
        features['obv_norm'] = self._normalize(df['obv'])
        features['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 5. Price Dynamics
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['volatility'] = features['log_returns'].rolling(20).std()
        
        return features.fillna(0)

    def _normalize(self, series):
        """Simple rolling normalization."""
        return (series - series.rolling(50).mean()) / (series.rolling(50).std() + 1e-8)
