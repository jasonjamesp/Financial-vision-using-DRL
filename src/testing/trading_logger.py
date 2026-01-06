import sqlite3
from datetime import datetime
import os
from pathlib import Path

class TradingLogger:
    def __init__(self, db_path="data/logs/trading_log.db"):
        self.db_path = Path(db_path)
        os.makedirs(self.db_path.parent, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                asset TEXT,
                action TEXT,
                price REAL,
                balance REAL,
                shares REAL,
                net_worth REAL,
                confidence REAL
            )
        """)
        conn.commit()
        conn.close()

    def log_trade(self, asset, action, price, balance, shares, net_worth, confidence=0.0):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trading_log 
            (asset, action, price, balance, shares, net_worth, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (asset, action, price, balance, shares, net_worth, confidence))
        conn.commit()
        conn.close()

    def get_trades(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trading_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_performance(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                MIN(net_worth) as min_worth,
                MAX(net_worth) as max_worth,
                (MAX(net_worth) - MIN(net_worth)) / MIN(net_worth) * 100 as total_return_pct
            FROM trading_log
        """)
        stats = dict(cursor.fetchone())
        conn.close()
        return stats
