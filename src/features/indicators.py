import pandas as pd
import numpy as np

class IndicatorEngine:
    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = IndicatorEngine.add_rsi(df)
        df = IndicatorEngine.add_macd(df)
        df = IndicatorEngine.add_bollinger_bands(df)
        df = IndicatorEngine.add_atr(df)
        df = IndicatorEngine.add_obv(df)
        df = IndicatorEngine.add_adx(df)
        df = IndicatorEngine.add_stochastic(df)
        df = IndicatorEngine.add_ema(df, [9, 21, 50])
        return df

    @staticmethod
    def add_rsi(df, period=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def add_bollinger_bands(df, period=20, std_dev=2):
        df['bb_mid'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_mid'] + (std * std_dev)
        df['bb_lower'] = df['bb_mid'] - (std * std_dev)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_percent'] = df['bb_percent'].fillna(0.5)
        return df

    @staticmethod
    def add_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        df['atr'] = df['atr'].fillna(method='bfill')
        return df

    @staticmethod
    def add_obv(df):
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return df

    @staticmethod
    def add_adx(df, period=14):
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = IndicatorEngine.add_atr(df, period)['atr'] # Reuse ATR logic for TR
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (np.abs(minus_dm).ewm(alpha=1/period).mean() / tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/period).mean()
        df['adx'] = df['adx'].fillna(0)
        return df

    @staticmethod
    def add_stochastic(df, period=14, k_period=3):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=k_period).mean()
        df['stoch_k'] = df['stoch_k'].fillna(50)
        df['stoch_d'] = df['stoch_d'].fillna(50)
        return df

    @staticmethod
    def add_ema(df, periods):
        for p in periods:
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        return df
