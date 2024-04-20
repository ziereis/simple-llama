import torch
import argparse
import numpy as np
from sentencepiece import SentencePieceProcessor
from qllama import QLLama
from llama_utils import load_tokenizer

# Load tokenizer
tokenizer = load_tokenizer("bin/tokenizer.model")

def generate_greedy(llama: QLLama, prompt: str, max_toks: int = 30) -> str:
    input_tokens = tokenizer.encode(prompt)
    output_tokens = []
    for token in input_tokens:
        _ = llama.forward(token, len(output_tokens))
        output_tokens.append(token)

    while len(output_tokens) < max_toks:
        latest_token = output_tokens[-1]
        out = llama.forward(latest_token, len(output_tokens))
        next_token = int(np.argmax(out))
        if next_token == tokenizer.eos_id():
            break
        output_tokens.append(next_token)
    return tokenizer.decode(output_tokens)

def generate_top_p(llama: QLLama, prompt: str, max_toks: int = 30, temperature: float = 1.0, top_p: float = 0.9) -> str:
    input_tokens = tokenizer.encode(prompt)
    output_tokens = []
    for token in input_tokens:
        _ = llama.forward(token, len(output_tokens))
        output_tokens.append(token)

    while len(output_tokens) < max_toks:
        latest_token = output_tokens[-1]
        out = llama.forward(latest_token, len(output_tokens))
        out = torch.tensor(out)
        probs = torch.softmax(out / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
        if next_token == tokenizer.eos_id():
            break
        output_tokens.append(next_token)
    return tokenizer.decode(output_tokens)

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, descending=True)
    probs_cumsum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_cumsum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort /= probs_sort.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token).item()

def run_multiple_configurations(llama, prompt, max_toks):
    print(f"Prompt: '{prompt}'")
    print("Greedy:", generate_greedy(llama, prompt, max_toks))
    temperatures = [0.1, 1.0, 2.0]
    top_ps = [0.1, 0.5, 0.95]
    for tp in top_ps:
        for temp in temperatures:
            output = generate_top_p(llama, prompt, max_toks, temperature=temp, top_p=tp)
            print(f"Top-p (temp={temp}, top_p={tp}): {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, help="path to exported llama f32 weights", default="bin/llama_q8.bin")
    parser.add_argument("--max-toks", type=int, help="max tokens to generate", default=30)
    parser.add_argument("prompt", type=str, nargs='*', help="prompt to generate from")
    args = parser.parse_args()

    rt = QLLama(args.bin)
    full_prompt = ' '.join(args.prompt) if args.prompt else None
    run_multiple_configurations(rt, full_prompt, args.max_toks)
