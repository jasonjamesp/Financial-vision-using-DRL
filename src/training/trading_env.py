import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.data_pipeline.gaf_encoder import GAFEncoder
from src.training.reward_function import RewardFunction
from src.utils.config import GAF_WINDOW_SIZE, IMAGE_SIZE

class TradingEnv(gym.Env):
    """
    Custom Environment for Financial Trading using GAF images.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, asset_name="ETH-USD", initial_balance=10000.0, transaction_fee=0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index()
        self.asset_name = asset_name
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        self.gaf_encoder = GAFEncoder()
        self.reward_fn = RewardFunction()
        
        # Action space: [Hold, Buy 25%, Buy 50%, Buy 100%, Sell 25%, Sell 50%, Sell 100%]
        # For simplicity in initial implementation: [0: Hold, 1: Buy All, 2: Sell All]
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 
        # 1. GAF Image (64, 64)
        # 2. Current Position State (Balance, Shares, Portfolio Value)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=-1, high=1, shape=(1, *IMAGE_SIZE), dtype=np.float32),
            "state": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        })
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = GAF_WINDOW_SIZE
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Window of data for GAF
        window = self.df.iloc[self.current_step-GAF_WINDOW_SIZE:self.current_step]['close'].values
        image = self.gaf_encoder.encode(window)
        image = image.reshape(1, *IMAGE_SIZE)
        
        state = np.array([
            self.balance,
            self.shares_held,
            self.net_worth
        ], dtype=np.float32)
        
        return {"image": image, "state": state}

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
        elif action == 2: # Sell All
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * (1 - self.transaction_fee)
                self.shares_held = 0
                trade_executed = True
        
        self.current_step += 1
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        current_return = (self.net_worth - prev_net_worth) / prev_net_worth
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        
        reward = self.reward_fn.calculate(
            current_return, 
            total_return, 
            drawdown, 
            trade_executed=trade_executed,
            transaction_costs=self.transaction_fee if trade_executed else 0.0
        )
        
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.4f}")
