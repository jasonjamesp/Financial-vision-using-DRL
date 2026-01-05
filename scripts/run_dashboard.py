import uvicorn
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print("Launching Financial Vision Dashboard...")
    print("Access at: http://localhost:8000")
    uvicorn.run("src.dashboard.app:app", host="0.0.0.0", port=8000, reload=True)
