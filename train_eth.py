import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from gatekeeper_env import FinancialVisionEnv

def train_eth():
    # --- Setup Directories ---
    log_dir = os.path.join('data', 'logs')
    models_dir = os.path.join('models')
    checkpoints_dir = os.path.join(models_dir, 'checkpoints')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print("--- Phase 3: ETH-USD Cross-Market Training ---")
    
    # --- 1. Environment Setup ---
    # CRITICAL: Filter for Ticker ID 5 (ETH-USD)
    env = FinancialVisionEnv(data_dir='./data', ticker_id=5)
    
    # Wrap with Monitor for CSV logging of rewards
    env = Monitor(env, filename=os.path.join(log_dir, 'phase3_training_log.csv'))
    
    # --- 2. Model Architecture ---
    # PPO with MultiInputPolicy (for Dict observation space)
    # Hyperparameters requested: lr=0.0003, ent_coef=0.01, n_steps=2048
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=0.0003,
        ent_coef=0.01,
        n_steps=2048,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )
    
    # --- 3. Callbacks ---
    # Save checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoints_dir,
        name_prefix='eth_gatekeeper'
    )
    
    # --- 4. Execution ---
    total_timesteps = 100000
    print(f"Starting training for {total_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving current model...")
    
    # --- 5. Save Final Model ---
    final_path = os.path.join(models_dir, 'eth_gatekeeper_final.zip')
    model.save(final_path)
    print(f"Training Complete. Final model saved to: {final_path}")

if __name__ == "__main__":
    train_eth()
