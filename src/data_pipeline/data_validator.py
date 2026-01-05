import pandas as pd
import numpy as np

class DataValidator:
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLCV data for consistency and gaps.
        """
        if df.empty:
            return df

        # Ensure columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        # Map case-insensitive
        df.columns = [c.lower() for c in df.columns]
        
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Missing column {col}")
                return pd.DataFrame()

        # 1. Consistency Check: High >= Open, Close, Low
        mask = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
        if not mask.all():
            print(f"Warning: Found {len(df) - mask.sum()} rows with inconsistent High prices. Correcting...")
            df.loc[~mask, 'high'] = df[['open', 'close', 'high']].max(axis=1)

        # 2. Consistency Check: Low <= Open, Close, High
        mask = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
        if not mask.all():
            print(f"Warning: Found {len(df) - mask.sum()} rows with inconsistent Low prices. Correcting...")
            df.loc[~mask, 'low'] = df[['open', 'close', 'low']].min(axis=1)

        # 3. Handle Missing Values
        if df.isnull().values.any():
            print(f"Warning: Found missing values. Forward-filling...")
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

        # 4. Remove Zero Volume (optional depending on strategy, but usually indicates bad data)
        # df = df[df['volume'] > 0]

        return df

    @staticmethod
    def detect_gaps(df: pd.DataFrame, expected_interval_mins: int = 15) -> list:
        """
        Detect gaps in time series data.
        """
        if df.empty:
            return []
            
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return []

        # Calculate time diffs
        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=expected_interval_mins)
        
        # Crypto is 24/7, Stocks have market hours (this is simple version)
        gaps = df[time_diffs > expected_diff].index.tolist()
        return gaps
