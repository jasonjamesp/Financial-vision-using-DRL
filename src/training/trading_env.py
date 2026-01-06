import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.data_pipeline.gaf_encoder import GAFEncoder
from src.training.reward_function import RewardFunction
from src.utils.config import GAF_WINDOW_SIZE, IMAGE_SIZE

class TradingEnv(gym.Env):
    """
    Enhanced Environment for Financial Trading with GAF + Technical Indicators.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, feature_df, asset_name="ETH-USD", initial_balance=10000.0, transaction_fee=0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.feature_df = feature_df.reset_index(drop=True)
        self.asset_name = asset_name
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        self.gaf_encoder = GAFEncoder()
        self.reward_fn = RewardFunction()
        
        # Action space: [0: Hold, 1: Buy All, 2: Sell All]
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 
        # 1. GAF Image (64, 64)
        # 2. Indicators (11 dims from FeatureBuilder)
        # 3. State (Balance, Shares, Portfolio Value, Drawdown, Profit/Loss Pct)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-1, high=1, shape=(1, *IMAGE_SIZE), dtype=np.float32),
            "indicators": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = GAF_WINDOW_SIZE
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.total_trades = 0
        self.successful_trades = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Window of data for GAF
        window = self.df.iloc[self.current_step-GAF_WINDOW_SIZE:self.current_step]['close'].values
        image = self.gaf_encoder.encode(window)
        image = image.reshape(1, *IMAGE_SIZE)
        
        # Latest indicator vector
        indicators = self.feature_df.iloc[self.current_step].values.astype(np.float32)
        
        # Position state
        drawdown = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-8)
        pnl_pct = (self.net_worth - self.initial_balance) / (self.initial_balance + 1e-8)
        
        state = np.array([
            self.balance,
            self.shares_held,
            self.net_worth,
            drawdown,
            pnl_pct
        ], dtype=np.float32)
        
        return {"image": image, "indicators": indicators, "state": state}

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # Actions: 0=Hold, 1=Buy All, 2=Sell All
        trade_executed = False
        if action == 1: # Buy All
            if self.balance > 0:
                shares_to_buy = self.balance / (current_price * (1 + self.transaction_fee))
                self.shares_held += shares_to_buy
                self.balance = 0
                trade_executed = True
                self.total_trades += 1
        elif action == 2: # Sell All
            if self.shares_held > 0:
                sale_proceeds = self.shares_held * current_price * (1 - self.transaction_fee)
                if sale_proceeds > (self.initial_balance / self.total_trades if self.total_trades > 0 else 0): # Simplified success check
                    self.successful_trades += 1
                self.balance += sale_proceeds
                self.shares_held = 0
                trade_executed = True
        
        self.current_step += 1
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        drawdown = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-8)
        current_return = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-8)
        total_return = (self.net_worth - self.initial_balance) / (self.initial_balance + 1e-8)
        
        reward = self.reward_fn.calculate(
            current_return, 
            total_return, 
            drawdown, 
            trade_executed=trade_executed,
            transaction_costs=self.transaction_fee if trade_executed else 0.0
        )
        
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {
            "net_worth": self.net_worth,
            "trades": self.total_trades,
            "win_rate": self.successful_trades / (self.total_trades + 1e-8)
        }
        
        return self._get_observation(), reward, done, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.4f}")
