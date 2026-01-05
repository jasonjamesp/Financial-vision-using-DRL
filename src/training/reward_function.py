import numpy as np

class RewardFunction:
    """
    Risk-adjusted reward logic for financial reinforcement learning.
    Combines returns, volatility, and drawdown penalties.
    """
    def __init__(self, risk_free_rate=0.0):
        self.risk_free_rate = risk_free_rate

    def calculate(self, current_return, total_return, drawdown, trade_executed=False, transaction_costs=0.0):
        """
        current_return: Log return of the period
        total_return: Cumulative return
        drawdown: Current drawdown from peak
        trade_executed: Boolean, whether a trade was made
        transaction_costs: Cost of current trade
        """
        # 1. Base return reward
        reward = current_return
        
        # 2. Transaction cost penalty
        if trade_executed:
            reward -= transaction_costs

        # 3. Drawdown penalty (non-linear)
        # Higher drawdown results in exponentially larger penalty
        if drawdown > 0.05: # 5% drawdown
            reward -= (drawdown ** 2) * 5
        
        # 4. Long-term performance incentive
        # Small bonus for keeping the strategy profitable
        if total_return > 0:
            reward += 0.001
            
        return reward

    def sharpe_style_reward(self, returns_window):
        """
        Calculates a reward based on the Sharpe ratio of a window of recent returns.
        """
        if len(returns_window) < 2:
            return 0.0
            
        mean_ret = np.mean(returns_window)
        std_ret = np.std(returns_window)
        
        if std_ret < 1e-6:
            return mean_ret
            
        return mean_ret / std_ret
