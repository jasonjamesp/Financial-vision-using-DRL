import subprocess
import os
import sys
from pathlib import Path

def start_training():
    project_root = Path(__file__).parent.parent
    os.environ['PYTHONPATH'] = str(project_root)
    
    print("Launching GAF-PPO Training Subsystem...")
    cmd = [sys.executable, "src/training/train.py"]
    
    # Run in background
    process = subprocess.Popen(cmd, cwd=project_root)
    print(f"Training started with PID: {process.pid}")
    return process

if __name__ == "__main__":
    start_training()
