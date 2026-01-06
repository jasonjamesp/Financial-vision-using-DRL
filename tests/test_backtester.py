from src.training.backtester import Backtester
from src.training.trading_env import TradingEnv
from src.training.ppo_agent import PPOAgent
import pandas as pd
import numpy as np
import torch

def test_backtester():
    print("Testing Backtester...")
    # Mock data
    data = {
        'high': [100, 105, 110, 115, 120] * 20,
        'low': [90, 95, 100, 105, 110] * 20,
        'close': [95, 100, 105, 110, 115] * 20,
        'volume': [1000] * 100
    }
    df = pd.DataFrame(data)
    feature_df = pd.DataFrame(np.random.randn(100, 12))
    
    env = TradingEnv(df, feature_df)
    agent = PPOAgent()
    
    bt = Backtester(agent, env)
    result = bt.run()
    
    assert hasattr(result, 'equity_curve')
    assert len(result.equity_curve) > 0
    assert result.sharpe_ratio is not None
    assert result.sortino_ratio is not None
    assert len(result.drawdown_curve) == len(result.equity_curve)
    
    print("Backtester tests passed!")

if __name__ == "__main__":
    try:
        test_backtester()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
