from gatekeeper_env import FinancialVisionEnv
import numpy as np
import os

def verify_phase2():
    # Ensure logs directory exists
    log_dir = os.path.join('data', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, 'phase2_verification.txt')
    print(f"Starting Phase 2 Verification. Logging to {log_file_path}...")
    
    # Init Env
    env = FinancialVisionEnv(data_dir='./data')
    obs, info = env.reset()
    
    with open(log_file_path, 'w') as log_file:
        for i in range(500):
            # Agent takes random action: 0=Hold, 1=Buy, 2=Sell
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            explanation = info.get('explanation', '')
            chips = info.get('chips', 0)
            vol_flag = info.get('volatility_flag', -1)
            
            # CRITICAL OUTPUT: Record ONLY "BLOCKED" events
            if "BLOCKED" in explanation:
                # Format: "Step [X]: Action [BUY] BLOCKED by [Rule Name] | Volatility_Flag=[0/1] | Chips=[X/3]"
                
                # Extract Action Name
                action_name = "BUY" if action == 1 else "SELL" if action == 2 else "HOLD"
                
                # Check Explanation to guess Rule Name (Parsing/Regex optional but simple string check works)
                rule_name = "Unknown"
                if "Volatility" in explanation:
                    rule_name = "Volatility Circuit Breaker"
                elif "Pattern" in explanation:
                    rule_name = "Pattern Gatekeeper"
                elif "Max Chips" in explanation:
                    rule_name = "Three-Chip Limit"
                
                log_line = (f"Step [{i}]: Action [{action_name}] BLOCKED by [{rule_name}] | "
                            f"Volatility_Flag=[{vol_flag}] | Chips=[{chips}/3]\n")
                
                log_file.write(log_line)
                # print(log_line.strip()) # Optional print
            
            if terminated:
                obs, info = env.reset()

    print("Verification complete.")

if __name__ == "__main__":
    verify_phase2()
