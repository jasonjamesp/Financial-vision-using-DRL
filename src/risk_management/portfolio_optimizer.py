import numpy as np
import pandas as pd

class PortfolioOptimizer:
    """
    Optimizes asset allocation across the target universe.
    """
    def __init__(self, assets):
        self.assets = assets

    def calculate_weights(self, returns_df):
        """
        Calculate weights using a simple Equal Weighted or Risk-Parity approach.
        returns_df: DataFrame of returns for each asset
        """
        if returns_df.empty:
            return {asset: 1.0/len(self.assets) for asset in self.assets}
            
        # Example: Inverse Volatility Weighting
        volatilities = returns_df.std()
        # Handle zero vol
        volatilities = volatilities.replace(0, 1e-6)
        
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        return weights.to_dict()

    def rebalance(self, current_positions, target_weights, total_value):
        """
        Calculate necessary trades to reach target weights.
        """
        trades = []
        for asset, weight in target_weights.items():
            target_value = total_value * weight
            current_value = current_positions.get(asset, 0)
            diff = target_value - current_value
            
            if abs(diff) > (total_value * 0.01): # 1% threshold for rebalancing
                trades.append({
                    "asset": asset,
                    "action": "BUY" if diff > 0 else "SELL",
                    "amount": abs(diff)
                })
        return trades
