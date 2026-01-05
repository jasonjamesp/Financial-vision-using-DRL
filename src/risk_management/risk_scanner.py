import numpy as np
import pandas as pd
from src.utils.config import MAX_DRAWDOWN_THRESHOLD, MIN_SHARPE_THRESHOLD

class RiskScanner:
    """
    Constant risk scanner to monitor portfolio health and trigger rebalancing.
    """
    def __init__(self, drawdown_threshold=MAX_DRAWDOWN_THRESHOLD, sharpe_threshold=MIN_SHARPE_THRESHOLD):
        self.drawdown_threshold = drawdown_threshold
        self.sharpe_threshold = sharpe_threshold

    def scan(self, portfolio_history):
        """
        Scan portfolio performance and return risk alerts.
        portfolio_history: List of net worth values
        """
        if len(portfolio_history) < 20:
            return []

        alerts = []
        history = np.array(portfolio_history)
        
        # 1. Calculate Max Drawdown
        peak = np.maximum.accumulate(history)
        drawdown = (peak - history) / peak
        current_drawdown = drawdown[-1]
        
        if current_drawdown > self.drawdown_threshold:
            alerts.append({
                "type": "DRAWDOWN_ALERT",
                "severity": "HIGH",
                "value": current_drawdown,
                "threshold": self.drawdown_threshold,
                "message": f"Critical drawdown detected: {current_drawdown:.2%}"
            })

        # 2. Calculate Rolling Sharpe Ratio (assume daily-like steps for demo)
        returns = np.diff(history) / history[:-1]
        if len(returns) >= 20:
            std = np.std(returns)
            if std > 0:
                sharpe = np.mean(returns) / std * np.sqrt(252) # Annualized
                if sharpe < self.sharpe_threshold:
                    alerts.append({
                        "type": "PERFORMANCE_ALERT",
                        "severity": "MEDIUM",
                        "value": sharpe,
                        "threshold": self.sharpe_threshold,
                        "message": f"Low Sharpe ratio detected: {sharpe:.2f}"
                    })

        return alerts

    def get_risk_adjusted_position_size(self, base_size, volatility):
        """
        Adjust position size based on asset volatility (Risk Parity style).
        """
        # Simple inverse volatility scaling
        if volatility == 0:
            return base_size
        
        adjustment = 1.0 / (volatility + 1e-6)
        return base_size * min(adjustment, 1.2) # Cap at 120% of base
