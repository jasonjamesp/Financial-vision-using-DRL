import numpy as np
import pandas as pd
from datetime import datetime
from src.training.backtest_result import BacktestResult, Trade

class Backtester:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def _get_date_str(self, df, idx):
        """Safely get date string from df."""
        if 'Date' in df.columns:
            return str(df.iloc[idx]['Date'])
        elif 'date' in df.columns:
            return str(df.iloc[idx]['date'])
        elif hasattr(df.index, 'strftime'):
            return str(df.index[idx])
        else:
            return "N/A"

    def run(self) -> BacktestResult:
        """
        Runs the agent through the environment and collects metrics.
        """
        obs, _ = self.env.reset()
        done = False
        equity_curve = [self.env.initial_balance]
        trades = []
        
        while not done:
            action, _ = self.agent.select_action(
                obs['image'], 
                obs['indicators'], 
                obs['state']
            )
            
            obs, reward, done, truncated, info = self.env.step(action)
            
            equity_curve.append(info['net_worth'])
            
            if action != 0:
                trades.append({
                    "timestamp": datetime.now().isoformat(),
                    "asset": self.env.asset_name,
                    "action": "BUY" if action == 1 else "SELL",
                    "price": float(self.env.df.iloc[self.env.current_step-1]['close']),
                    "net_worth": float(info['net_worth'])
                })
        
        # Calculate Metrics
        returns = pd.Series(equity_curve).pct_change().dropna()
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Sharpe Ratio
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252 * 24)
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / (downside_returns.std() + 1e-8)) * np.sqrt(252 * 24)
        
        # Drawdown Curve
        peak = pd.Series(equity_curve).expanding().max()
        drawdown_curve = (peak - equity_curve) / (peak + 1e-8)
        max_drawdown = drawdown_curve.max()
        
        # Price Series (last window)
        price_series = [float(p) for p in self.env.df['close'].values]
        
        return BacktestResult(
            asset=self.env.asset_name,
            start_date=self._get_date_str(self.env.df, 0),
            end_date=self._get_date_str(self.env.df, -1),
            initial_balance=self.env.initial_balance,
            final_balance=equity_curve[-1],
            total_return_pct=float(total_return * 100),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown_pct=float(max_drawdown * 100),
            win_rate=float(info['win_rate']),
            total_trades=int(info['trades']),
            equity_curve=[float(x) for x in equity_curve],
            drawdown_curve=[float(x) for x in drawdown_curve],
            price_series=price_series,
            trades=trades
        )

