import subprocess
import sys
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("Launching Financial Fraud Analytics Dashboard...")
    print("Local URL: http://localhost:8501")
    print("=" * 60)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

