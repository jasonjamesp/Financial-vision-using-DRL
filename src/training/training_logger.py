import sqlite3
from datetime import datetime
import os
from pathlib import Path

class TrainingLogger:
    def __init__(self, db_path="data/logs/training_log.db"):
        self.db_path = Path(db_path)
        os.makedirs(self.db_path.parent, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                episode INTEGER,
                reward REAL,
                net_worth REAL,
                actor_loss REAL,
                critic_loss REAL,
                trades INTEGER,
                data_source TEXT,
                data_checksum TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_episode(self, episode, reward, net_worth, actor_loss, critic_loss, trades, data_source="cache", data_checksum=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO training_log 
            (episode, reward, net_worth, actor_loss, critic_loss, trades, data_source, data_checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (episode, reward, net_worth, actor_loss, critic_loss, trades, data_source, data_checksum))
        conn.commit()
        conn.close()

    def get_diary(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training_log ORDER BY episode DESC LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_episodes,
                AVG(reward) as avg_reward,
                MAX(net_worth) as max_net_worth,
                AVG(actor_loss) as avg_actor_loss,
                AVG(critic_loss) as avg_critic_loss
            FROM training_log
        """)
        stats = dict(cursor.fetchone())
        conn.close()
        return stats
