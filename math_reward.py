"""Example Miles-compatible reward function for math reasoning.

Upload this via: POST /v1/reward_functions
Miles loads it via --custom-rm-path reward_fn.reward

The function must be async and accept (args, sample, **kwargs) -> float.
sample.response contains the model completion, sample.label contains ground truth.
"""

import re


async def reward(args, sample, **kwargs) -> float:
    """Reward = 0.5 * format_correct + 0.5 * answer_correct."""
    response = sample.response
    label = sample.label or ""

    # Check format: model should wrap answer in <answer>...</answer> tags
    format_ok = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))

    # Check correctness: ground truth appears in the answer tags
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    correct = False
    if answer_match and label:
        correct = label.strip() in answer_match.group(1).strip()

    return 0.5 * float(format_ok) + 0.5 * float(correct)
