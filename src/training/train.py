import torch
import numpy as np
import pandas as pd
from src.training.trading_env import TradingEnv
from src.training.ppo_agent import PPOAgent
from src.data_pipeline.data_manager import DataManager
from src.data_pipeline.data_validator import DataValidator
from src.utils.config import EPISODES, CHECKPOINT_INTERVAL, CHECKPOINT_DIR
import os

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Load Data
    dm = DataManager()
    raw_df = dm.fetch_ohlcv("ETH-USD")
    
    dv = DataValidator()
    df = dv.validate_ohlcv(raw_df)
    
    if df.empty:
        print("Failed to load valid data. Exiting.")
        return

    # 2. Setup Environment
    env = TradingEnv(df)
    
    # 3. Setup Agent
    agent = PPOAgent()
    
    # 4. Training Loop
    print(f"Starting training for {EPISODES} episodes...")
    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, log_prob = agent.select_action(obs['image'], obs['state'])
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Here you would typically store transitions in a buffer
            # and call agent.update() after a certain number of steps
            
            obs = next_obs
            total_reward += reward
            
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Net Worth: {env.net_worth:.2f}")
            
        if episode % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = CHECKPOINT_DIR / f"ppo_eth_usd_{episode}.pth"
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    train()
