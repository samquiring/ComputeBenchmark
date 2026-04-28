import re
from datasets import load_dataset
from transformers import PreTrainedTokenizer

SYSTEM_PROMPT = "You are a helpful assistant. Solve math problems step by step."


def load_gsm8k(split: str = "train"):
    return load_dataset("gsm8k", "main", split=split)


def extract_answer(text: str) -> str | None:
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    return m.group(1).replace(",", "").strip() if m else None


def make_prompt(question: str, tokenizer: PreTrainedTokenizer, enable_thinking: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3 models support an enable_thinking flag; ignored by other tokenizers
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_dataset(tokenizer: PreTrainedTokenizer, split: str = "train"):
    raw = load_gsm8k(split)

    def process(example):
        return {
            "prompt": make_prompt(example["question"], tokenizer),
            "answer": extract_answer(example["answer"]),
            "question": example["question"],
        }

    return raw.map(process, remove_columns=raw.column_names)
