import os
import subprocess
import sys

def run_vllm_server():
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/rds/models/gpt-oss-120b",
        "--served-model-name", "gpt-oss-120b",
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--port", "8010",
        "--tokenizer", "/rds/models/gpt-oss-120b",
        
        # --- NEW SETTING ---
        # Controls how much of the GPU is strictly reserved for vLLM.
        # 0.90 = 90% (Default). 
        # 0.95 = 95% (Maximizes context length).
        "--gpu-memory-utilization", "0.8", 
        # -------------------
    ]

    print(f"Starting vLLM server with command:\n{' '.join(cmd)}\n")
    print("Logs will appear below (Press Ctrl+C to stop):")
    print("-" * 50)

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nServer crashed with error code {e.returncode}")

if __name__ == "__main__":
    run_vllm_server()