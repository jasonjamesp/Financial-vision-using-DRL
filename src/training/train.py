from src.features.indicators import IndicatorEngine
from src.features.feature_builder import FeatureBuilder

def train():
    config = load_config()
    checkpoint_dir = Path("models/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = TrainingLogger()
    
    # 1. Load Data
    dm = DataManager()
    raw_df = dm.fetch_ohlcv(config.get("asset", "ETH-USD"), days=config.get("data_days", 59))
    
    dv = DataValidator()
    df = dv.validate_ohlcv(raw_df)
    
    if df.empty:
        print("Failed to load valid data. Exiting.")
        return

    # 2. Extract Features
    df = IndicatorEngine.compute_all(df)
    fb = FeatureBuilder()
    feature_df = fb.build_features(df)
    
    data_checksum = get_data_checksum(df)
    print(f"Data verified. Checksum: {data_checksum}")

    # 3. Setup Environment
    env = TradingEnv(df, feature_df, asset_name=config.get("asset", "ETH-USD"))
    
    # 4. Setup Agent
    agent = PPOAgent()
    
    # 5. Training Loop
    num_episodes = config.get("episodes", 500000)
    print(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        # Hot-reload config every 10 episodes
        if episode % 10 == 0:
            try:
                config = load_config()
            except Exception as e:
                print(f"Warning: Failed to hot-reload config: {e}")

        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, log_prob = agent.select_action(
                obs['image'], 
                obs['indicators'], 
                obs['state']
            )
            next_obs, reward, done, truncated, info = env.step(action)
            
            # PPO update logic would go here
            
            obs = next_obs
            total_reward += reward
        
        # Log to SQLite
        logger.log_episode(
            episode=episode,
            reward=total_reward,
            net_worth=env.net_worth,
            actor_loss=0.0, 
            critic_loss=0.0,
            trades=info.get('trades', 0),
            data_source="cache",
            data_checksum=data_checksum
        )
            
        if episode % config.get("log_interval", 10) == 0:
            print(f"Episode {episode}, Reward: {total_reward:.4f}, Worth: {env.net_worth:.2f}, Win Rate: {info.get('win_rate', 0):.2%}")
            
        if episode % config.get("checkpoint_interval", 10000) == 0:
            checkpoint_path = checkpoint_dir / f"ppo_{config.get('asset', 'eth_usd')}_{episode}.pth"
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    train()
