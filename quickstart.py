#!/usr/bin/env python3
"""
Veri Quickstart: Train a math reasoning model with GRPO

This script walks through the full Veri training workflow:
  1. Connect the GSM8K dataset from Hugging Face
  2. Upload a reward function that scores math reasoning
  3. Submit a GRPO training job
  4. Wait for completion and download the checkpoint
  5. Run inference on the fine-tuned model

Veri handles GPU provisioning, job orchestration, and checkpoint delivery.
You just provide the dataset, reward function, and hyperparameters.

Prerequisites:
  pip install veri-sdk
  veri login --key vk_your_api_key

Usage:
  python quickstart.py
"""

from veri_sdk import Client

# ─────────────────────────────────────────────────────────────────────
# 1. Connect to Veri
# ─────────────────────────────────────────────────────────────────────
# The client reads your API key from ~/.veri/credentials.json
# (created by `veri login`) or you can pass it directly.

client = Client()
# client = Client(api_key="vk_...", base_url="https://api.veri.studio")

print("Connected to Veri API\n")


# ─────────────────────────────────────────────────────────────────────
# 2. Connect the GSM8K dataset
# ─────────────────────────────────────────────────────────────────────
# GSM8K is OpenAI's dataset of 8,800 grade school math problems with
# step-by-step solutions. It's the standard benchmark for training
# reasoning models (used by DeepSeek R1, Qwen, etc.)
#
# Veri pulls directly from Hugging Face — no local download needed.
# The column_mapping tells Veri which fields to use:
#   "question" → becomes the training prompt
#   "answer"   → becomes the label your reward function checks against

dataset = client.datasets.connect(
    name="gsm8k-train",
    source_type="hf",
    huggingface_dataset="openai/gsm8k",
    huggingface_config={
        "split": "train",
        "column_mapping": {
            "question": "prompt",
            "answer": "label",
        },
    },
)
print(f"Dataset: {dataset.id} ({dataset.name})\n")


# ─────────────────────────────────────────────────────────────────────
# 3. Upload the reward function
# ─────────────────────────────────────────────────────────────────────
# The reward function scores each model completion. GRPO generates
# multiple completions per prompt, scores them, and reinforces the
# ones that score above the group average.
#
# This reward checks two things:
#   - Did the model use <reasoning>...</reasoning><answer>...</answer> format?
#   - Did it get the correct final answer?
#
# The TRL format expects:
#   def reward(completions, answer, **kwargs) -> list[float]
#
# See examples/math_reward_trl.py for the full implementation.

reward_fn = client.reward_functions.upload(
    "examples/math_reward_trl.py",
    format="trl",
)
print(f"Reward function: {reward_fn.id} ({reward_fn.name})\n")


# ─────────────────────────────────────────────────────────────────────
# 4. Submit the training job
# ─────────────────────────────────────────────────────────────────────
# GRPO (Group Relative Policy Optimization) is the algorithm behind
# DeepSeek R1. For each prompt, the model generates `rollouts_per_prompt`
# completions, your reward function scores each one, and the model
# learns to produce better completions over time.
#
# Key hyperparameters:
#   learning_rate       — How fast the model updates. 1e-6 is stable.
#   rollouts_per_prompt — More rollouts = better advantage estimates,
#                         but costs more compute. 8 is a good default.
#   max_response_length — Token cap per completion. Keep low to save VRAM.
#   kl_coef             — Prevents the model from drifting too far from
#                         the original weights. 0.001 is conservative.
#
# GPU selection:
#   Qwen2.5-7B fits on a single A100-80GB. For larger models (14B+),
#   increase gpu_count or use H100s.

job = client.training_jobs.create(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    dataset_id=dataset.id,
    reward_function_id=reward_fn.id,
    output_name="qwen2.5-7b-math-reasoning",
    method="grpo",
    hyperparameters={
        "learning_rate": 1e-6,
        "rollouts_per_prompt": 8,
        "max_response_length": 2048,
        "kl_coef": 0.001,
        "num_epochs": 1,
    },
    gpu_type="A100-80GB",
    gpu_count=1,
)
print(f"Job submitted: {job.id}")
print(f"Status: {job.status}")
print(f"Dashboard: {job.dashboard_url or 'available once running'}\n")


# ─────────────────────────────────────────────────────────────────────
# 5. Wait for completion
# ─────────────────────────────────────────────────────────────────────
# job.wait() polls the API every 10 seconds until the job reaches
# a terminal state (completed, failed, or cancelled).
#
# Training time depends on dataset size and GPU:
#   ~8K prompts × 8 rollouts × 1 epoch ≈ 1-3 hours on A100-80GB

print("Waiting for training to complete (this may take a while)...")
job.wait(poll_interval=30)

if job.status == "completed":
    print(f"\nTraining complete!")
    print(f"Cost: ${job.cost_usd:.4f}" if job.cost_usd else "")
    print(f"Checkpoint: {job.download_url}")
elif job.status == "failed":
    print(f"\nTraining failed: {job.error_message}")
    exit(1)


# ─────────────────────────────────────────────────────────────────────
# 6. Load and test the fine-tuned model
# ─────────────────────────────────────────────────────────────────────
# The checkpoint is a standard HuggingFace model. Load it with
# transformers and run inference. After GRPO training on GSM8K,
# the model should produce structured reasoning like:
#
#   <reasoning>
#   Natalia sold 48 clips in April.
#   In May, she sold half: 48 / 2 = 24.
#   Total: 48 + 24 = 72
#   </reasoning>
#   <answer>72</answer>

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nDownloading and loading checkpoint...")
    path = job.download(output_dir="./checkpoints")

    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Test with a GSM8K-style problem
    prompt = (
        "Natalia sold clips to 48 of her friends in April, and then "
        "she sold half as many clips in May. How many clips did "
        "Natalia sell altogether in April and May?"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"\nResponse:\n{response}")

except ImportError:
    print("\nInstall transformers to load the checkpoint locally:")
    print("  pip install transformers torch")
    print(f"\nOr download directly: {job.download_url}")


# ─────────────────────────────────────────────────────────────────────
# What's next?
# ─────────────────────────────────────────────────────────────────────
#
# Improve the reward function:
#   - Add partial credit for intermediate steps
#   - Penalize overly long responses
#   - Reward specific mathematical notation
#
# Try different models:
#   - Qwen/Qwen2.5-1.5B-Instruct  → fast experimentation
#   - Qwen/Qwen2.5-7B-Instruct    → good quality/cost balance
#   - meta-llama/Llama-3.1-8B-Instruct → strong reasoning baseline
#   - google/gemma-4-E4B-it        → multimodal, 128K context
#
# Try video gen fine-tuning:
#   job = client.training_jobs.create(
#       base_model="THUDM/CogVideoX-2b",
#       dataset_id="ds_vid123",
#       method="sft_video_gen",
#       output_name="cogvideo-custom",
#       hyperparameters={"lora_rank": 64, "num_epochs": 30},
#       gpu_type="A100-80GB",
#       gpu_count=1,
#   )
#
# Docs: https://docs.veri.studio
# API Reference: https://docs.veri.studio/api-reference/introduction
# Discord: https://discord.gg/DBZSRzPAuR
