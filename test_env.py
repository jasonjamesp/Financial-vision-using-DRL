from gatekeeper_env import FinancialVisionEnv
import numpy as np

def test_environment():
    print("Initializing Environment...")
    # Initialize without data files to trigger dummy data generation
    env = FinancialVisionEnv(data_dir='./dummy_path')
    
    obs, info = env.reset()
    print("Initial Info:", info)
    print("Obs keys:", obs.keys())
    print("Image shape:", obs['image'].shape)
    
    print("\nStarting Test Loop (20 steps)...")
    
    # We will try to force some scenarios if we can, or just inspect the logs
    # Since we can't easily force the internal random state from here without hacking,
    # we'll just run enough steps and print the 'explanation' when it's interesting.
    
    for i in range(20):
        # Pick a random action: 0=Hold, 1=Buy, 2=Sell
        action = np.random.choice([0, 1, 2])
        
        # Hack to test chips limit: Always buy if we can
        # action = 1 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        explanation = info.get('explanation', '')
        chips = info.get('chips', 0)
        
        # Only print interesting steps
        if "BLOCKED" in explanation or "BUY" in explanation or "SELL" in explanation:
            print(f"Step {i}: Action={action} | Chips={chips} | Reward={reward:.2f}")
            print(f"   Log: {explanation}")
            
        if terminated:
            print("Terminated.")
            break

    print("\nTest Complete.")

if __name__ == "__main__":
    test_environment()
