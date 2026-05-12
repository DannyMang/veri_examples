"""TRL-compatible reward function for math reasoning.

Expected signature for TRL GRPOTrainer:
    def reward(completions, answer, **kwargs) -> list[float]

- completions: list of strings (model completions)
- answer: list of strings (ground truth, passed from dataset column)
- returns: list of floats (one score per completion)
"""

import re


def reward(completions, answer, **kwargs):
    scores = []
    for completion, expected in zip(completions, answer):
        # Handle chat-format completions (list of message dicts)
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        # Check format: answer tags present
        format_ok = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))

        # Check correctness: expected answer appears in answer tags
        correct = False
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match and expected:
            correct = expected.strip() in match.group(1).strip()

        # Fallback: check if the answer appears anywhere in the response
        if not correct and expected:
            correct = expected.strip() in text

        scores.append(0.5 * float(format_ok) + 0.5 * float(correct))

    return scores
