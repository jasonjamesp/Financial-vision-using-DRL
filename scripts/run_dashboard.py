import uvicorn
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("Launching GAF-PPO Dual Dashboard...")
    print("Dashboard Access: http://localhost:8000")
    uvicorn.run("src.dashboard.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
