import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from src.utils.config import CACHE_DIR, CANDLE_INTERVAL
import os

class DataManager:
    def __init__(self, cache_db="market_data.db"):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.conn = sqlite3.connect(CACHE_DIR / cache_db)
        self._create_table()

    def _create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS ohlcv (
            asset TEXT,
            timestamp DATETIME,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (asset, timestamp)
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def fetch_ohlcv(self, asset: str, interval: str = CANDLE_INTERVAL, days: int = 59) -> pd.DataFrame:
        """
        Fetch OHLCV data from yfinance and cache it. 
        yfinance has a 60-day limit for 15m data.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            print(f"Fetching {asset} for last {days} days...")
            df = yf.download(asset, start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print(f"No data found for {asset}")
                return self.get_cached_data(asset)

            # Flatten columns if MultiIndex (yf download sometimes returns it)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)
            df.rename(columns={'Datetime': 'timestamp', 'Date': 'timestamp'}, inplace=True)
            
            # Format for SQLite
            df['asset'] = asset
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Upsert into cache
            df[['asset', 'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].to_sql(
                'ohlcv', self.conn, if_exists='append', index=False, method=self._upsert
            )
            self.conn.commit()
            
            return self.get_cached_data(asset)
            
        except Exception as e:
            print(f"Error fetching {asset}: {e}")
            return self.get_cached_data(asset)

    def _upsert(self, table, conn, keys, data_iter):
        """Custom upsert method for to_sql"""
        from sqlite3 import IntegrityError
        query = f"""
            INSERT INTO {table.name} ({', '.join(keys)})
            VALUES ({', '.join(['?'] * len(keys))})
            ON CONFLICT(asset, timestamp) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
        """
        conn.executemany(query, data_iter)

    def get_cached_data(self, asset: str) -> pd.DataFrame:
        query = f"SELECT * FROM ohlcv WHERE asset = ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, self.conn, params=(asset,))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

if __name__ == "__main__":
    dm = DataManager()
    # Test with ETH-USD
    data = dm.fetch_ohlcv("ETH-USD")
    print(f"ETH-USD data head:\n{data.head()}")
    print(f"Total rows for ETH-USD: {len(data)}")
