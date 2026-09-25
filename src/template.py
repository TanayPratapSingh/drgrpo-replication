"""The paper's R1 prompt template and stop string, free of any framework import.

Shared by the MLX sampler (rollout.py) and the PyTorch/vLLM trainer, so both
backends prompt the model with byte identical text.
"""

STOP_STR = "</answer>"

# Template 1 of the paper, copied from apply_r1_template in train_zero_math.py.
R1_PREFIX = (
    "A conversation between User and Assistant. The User asks a question, and the "
    "Assistant solves it. The Assistant first thinks about the reasoning process in "
    "the mind and then provides the User with the answer. The reasoning process is "
    "enclosed within <think> </think> and answer is enclosed within <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think> <answer> answer "
    "here </answer>.\nUser: "
)


def r1_prompt(question: str) -> str:
    return R1_PREFIX + question + "\nAssistant: <think>"
