import json
from dataclasses import dataclass, asdict
from typing import List, Dict
from datetime import datetime

@dataclass
class Trade:
    timestamp: str
    asset: str
    action: str
    price: float
    net_worth: float

@dataclass
class BacktestResult:
    asset: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    equity_curve: List[float]
    drawdown_curve: List[float]
    price_series: List[float]
    trades: List[Dict]

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.to_json())
