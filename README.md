# veri-examples

Runnable example scripts for the [Veri](https://veri.studio) RL post-training platform.

## Contents

- `quickstart.py` — End-to-end GRPO training: connect a HuggingFace dataset, upload a reward function, submit a job, wait, download the checkpoint.
- `math_reward_trl.py` — TRL-format reward function for math reasoning (`def reward(completions, answer, **kwargs) -> list[float]`).
- `math_reward.py` — Miles-format async reward function (`async def reward(args, sample, **kwargs) -> float`).
- `gsm8k_prompts.jsonl` — 20-row sample dataset in Veri's chat format, useful for smoke tests without pulling the full GSM8K from HuggingFace.

## Run

```bash
pip install veri-sdk
veri login --key vk_your_key
python quickstart.py
```

## Docs

Full documentation: [docs.veri.studio](https://docs.veri.studio).
