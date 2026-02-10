"""
vLLM server for gpt-oss-120b (OpenAI open-weight reasoning model).

Model: openai/gpt-oss-120b
  - 117B total params, 5.1B active (MoE with 128 experts, 4 active)
  - MXFP4 quantized MoE weights -- fits on a single 80GB GPU
  - Uses OpenAI "harmony" response format with <|channel|>analysis / <|channel|>final
  - Natively supports: function calling, structured outputs, chain-of-thought

Configured for cmbagent:
  - Structured output  (response_format with Pydantic json_schema for formatter agents)
  - Tool calling       (executor_response_formatter, controller, plan_recorder, etc.)
  - Reasoning parsing  (separates analysis/final channels)

Prerequisites:
  OpenAI recommends a patched vLLM build for gpt-oss:

    uv pip install --pre vllm==0.10.1+gptoss \\
      --extra-index-url https://wheels.vllm.ai/gpt-oss/ \\
      --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \\
      --index-strategy unsafe-best-match

  If you are on that build, the explicit parser flags below should still
  work fine.  If you are on mainline vLLM >= 0.10, check that the
  seed_oss / openai_gptoss parsers are available (they were merged upstream).

Usage:
  python server.py
"""
import os
import subprocess
import sys


def run_vllm_server():
    env = os.environ.copy()

    # HuggingFace offline mode -- model must already be cached locally
    env["HF_HUB_OFFLINE"] = "1"

    # Pin to GPU 1
    env["CUDA_VISIBLE_DEVICES"] = "1"

    MODEL_PATH = "/rds/models/gpt-oss-120b"
    SERVED_NAME = "gpt-oss-120b"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",

        "--model", MODEL_PATH,
        "--served-model-name", SERVED_NAME,
        "--tokenizer", MODEL_PATH,

        "--host", "0.0.0.0",
        "--port", "8011",

        "--trust-remote-code",
        "--dtype", "auto",

        # Single GPU (CUDA_VISIBLE_DEVICES already selects which one)
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.95",

        # Use model's generation_config.json which includes <|call|> (token 200012)
        # as an EOS token -- required for tool calling to stop correctly.
        # Also sets do_sample=true and includes <|return|> + <|endoftext|> as EOS.
        # Set to "vllm" to ignore the model's config and use vLLM defaults.
        "--generation-config", "auto",

        # ---- Tool calling ----
        # Required for cmbagent agents that use register_function:
        #   executor_response_formatter, controller, plan_recorder,
        #   review_recorder, idea_saver, aas_keyword_finder, etc.
        # "openai" parser handles the harmony tool-call format:
        #   <|start|>assistant to=functions.<name><|channel|>commentary json<|message|>{args}<|call|>
        # NOTE: "seed_oss" is for Seed models (XML <seed:tool_call>), NOT gpt-oss.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "openai",

        # ---- Reasoning ----
        # gpt-oss uses harmony channels for chain-of-thought:
        #   <|channel|>analysis  = reasoning (not shown to user)
        #   <|channel|>final     = actual response content
        # openai_gptoss parser separates these in the API response.
        "--reasoning-parser", "openai_gptoss",

        # ---- Structured output (Pydantic response_format) ----
        # Needed for cmbagent formatter agents that set response_format to
        # Pydantic models (EngineerResponse, PlannerResponse, StructuredMarkdown, etc.)
        # "auto" backend picks xgrammar/outlines based on the request.
        # reasoning_parser tells guided decoding to only apply schema constraints
        # after reasoning ends (otherwise it would try to constrain the
        # analysis channel tokens to the JSON schema).
        "--structured-outputs-config.backend", "auto",
        "--structured-outputs-config.reasoning_parser", "openai_gptoss",
    ]

    print("Starting vLLM server with command:")
    print(" ".join(cmd))
    print()
    print("GPU: 1 (via CUDA_VISIBLE_DEVICES)")
    print("Port: 8011")
    print("-" * 60)

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except subprocess.CalledProcessError as e:
        print("\nServer exited with code " + str(e.returncode))


if __name__ == "__main__":
    run_vllm_server()
