import subprocess
import os
import sys
from pathlib import Path

def start_live_test():
    project_root = Path(__file__).parent.parent
    os.environ['PYTHONPATH'] = str(project_root)
    
    print("Launching GAF-PPO Live Testing Subsystem...")
    cmd = [sys.executable, "src/testing/live_test.py"]
    
    # Run in background
    process = subprocess.Popen(cmd, cwd=project_root)
    print(f"Live testing started with PID: {process.pid}")
    return process

if __name__ == "__main__":
    start_live_test()
