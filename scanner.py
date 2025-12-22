import os
import numpy as np
import time
from stable_baselines3 import PPO
from gatekeeper_env import FinancialVisionEnv

def run_scanner():
    print("--- Financial Vision-India: DAILY SCANNER ---")
    
    # 1. Paths
    model_path = os.path.join('models', 'eth_gatekeeper_final.zip')
    data_dir = './data'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return

    # 2. Load Data (Ticker 0 = RELIANCE.NS)
    # We manually look for the latest data point for Ticker 0
    # In a real system, this would call yfinance/processed live data.
    # Here we simulate by finding the last occurrence of Ticker 0 in our data file.
    
    try:
        metadata = np.load(os.path.join(data_dir, 'obs_metadata.npy'))
        images = np.load(os.path.join(data_dir, 'obs_images.npy'))
        
        # Filter for Ticker 0
        ticker_indices = np.where(metadata[:, 0] == 0)[0]
        
        if len(ticker_indices) == 0:
            print("Error: No data found for Ticker 0 (RELIANCE).")
            return
            
        # Get Latest Window
        latest_idx = ticker_indices[-1]
        
        # Prepare Observation
        latest_img = images[latest_idx]
        if len(latest_img.shape) == 2:
            latest_img = np.expand_dims(latest_img, axis=-1)
            
        latest_meta = metadata[latest_idx]
        # Metadata: [Ticker, Pattern, Volatility, Close]
        
        # 3. Manual Gatekeeper Logic
        # We must replicate the Env logic to inform the user WHY
        
        raw_pattern = int(latest_meta[1]) # -1, 0, 1
        vol_flag = int(latest_meta[2])    # 0=Lockdown, 1=Safe
        close_price = latest_meta[3]
        
        gatekeeper_status = "PASS"
        block_reason = ""
        
        # Rule 3: Volatility
        if vol_flag == 0:
            gatekeeper_status = "BLOCK"
            block_reason = "Volatility Lockdown"
            
        # Rule 1: Pattern
        elif raw_pattern == 0:
            gatekeeper_status = "BLOCK"
            block_reason = "No Whitelisted Pattern"
            
        # We cannot check Rule 2 (Chips) here easily without portfolio state, 
        # so we assume Chips=0 (Fresh Entry).
        
        # 4. AI Prediction
        ai_decision = "N/A"
        
        if gatekeeper_status == "PASS":
            print("Gatekeeper Status: PASS. Invoking AI Agent...")
            
            # Load Model
            model = PPO.load(model_path)
            
            # Construct Obs Dict
            # Obs must match Env space
            # Pattern in Obs is mapped (-1->0, 0->1, 1->2)
            pattern_obs = raw_pattern + 1
            
            obs = {
                "image": np.expand_dims(latest_img, axis=0), # Batch dim
                "chips": np.array([0]), # Batch dim, assume 0 chips
                "pattern": np.array([pattern_obs]) # Batch dim
            }
            
            action, _ = model.predict(obs)
            action = int(action[0])
            
            mp = {0: "HOLD", 1: "BUY", 2: "SELL"}
            ai_decision = mp.get(action, "UNKNOWN")
        else:
            ai_decision = f"BLOCKED ({block_reason})"

        # 5. Output
        current_date = time.strftime("%Y-%m-%d")
        print(f"[{current_date}] | Ticker: RELIANCE | Gatekeeper: {gatekeeper_status} | AI Decision: {ai_decision}")
        print(f"Details: Pattern={raw_pattern}, Volatility={vol_flag}, Price={close_price:.2f}")

    except Exception as e:
        print(f"Scanner Error: {e}")

if __name__ == "__main__":
    run_scanner()
