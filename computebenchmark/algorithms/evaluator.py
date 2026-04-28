import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.gsm8k import extract_answer, load_gsm8k, make_prompt


def evaluate_gsm8k(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    split: str = "test",
    max_samples: int | None = None,
    max_new_tokens: int = 512,
) -> dict:
    dataset = load_gsm8k(split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    model.eval()
    for example in tqdm(dataset, desc="GSM8K eval", leave=False):
        prompt = make_prompt(example["question"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if extract_answer(generated) == extract_answer(example["answer"]):
            correct += 1

    total = len(dataset)
    return {"gsm8k_accuracy": correct / total, "correct": correct, "total": total}


def evaluate_math(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_samples: int = 500,
    max_new_tokens: int = 1024,
) -> dict:
    dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))

    correct = 0
    model.eval()
    for example in tqdm(dataset, desc="MATH eval", leave=False):
        messages = [
            {"role": "system", "content": "Solve the math problem. Put your final answer inside \\boxed{}."},
            {"role": "user", "content": example["problem"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # extract \boxed{...} answer
        import re
        m = re.search(r"\\boxed\{([^}]+)\}", generated)
        predicted = m.group(1).strip() if m else None
        m2 = re.search(r"\\boxed\{([^}]+)\}", example["solution"])
        ground_truth = m2.group(1).strip() if m2 else None
        if predicted is not None and predicted == ground_truth:
            correct += 1

    total = len(dataset)
    return {"math_accuracy": correct / total, "correct": correct, "total": total}
