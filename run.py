import torch
from sentencepiece import SentencePieceProcessor
import argparse
import numpy as np
import os
from qllama import Runtime, CreateRuntime
from llama_utils import load_tokenizer

tokenizer = load_tokenizer("bin/tokenizer.model")

def generate_greedy(llama: Runtime, prompt: str, max_toks: int = 30) -> str:
  input_tokens = tokenizer.encode(prompt)
  output_tokens = []
  # feed the entire prompt as context
  for token in input_tokens:
    # _ skip, because we dont care about the next token predictions of the prompt. Just to populate KV Cache
    _ = llama.forward(token, len(output_tokens))
    output_tokens.append(token)
    os.system('clear')
    print(tokenizer.decode(output_tokens))

  # generate the rest of the tokens
  while len(output_tokens) < max_toks:
    latest_token = output_tokens[-1]
    out = llama.forward(latest_token, len(output_tokens))
    next_token = int(np.argmax(out))
    if next_token == tokenizer.eos_id():
      break
    output_tokens.append(next_token)
    os.system('clear')
    print(tokenizer.decode(output_tokens))
  os.system('clear')
  print(tokenizer.decode(output_tokens))
  return tokenizer.decode(output_tokens)

def generate_top_p(llama: Runtime, prompt: str, max_toks: int = 30, temperature: float = 0.1, top_p: float = 0.90) -> str:
  input_tokens = tokenizer.encode(prompt)
  output_tokens = []
  for token in input_tokens:
    _ = llama.forward(token, len(output_tokens))
    output_tokens.append(token)
    os.system('clear')
    print(tokenizer.decode(output_tokens))

  while len(output_tokens) < max_toks:
    latest_token = output_tokens[-1]
    out = llama.forward(latest_token, len(output_tokens))
    out = torch.tensor(out)
    probs = torch.softmax(out / temperature, dim=-1)
    next_token = sample_top_p(probs, top_p)
    if next_token == tokenizer.eos_id():
      break
    output_tokens.append(next_token)
    os.system('clear')
    print(tokenizer.decode(output_tokens))
  os.system('clear')
  print(tokenizer.decode(output_tokens))
  return tokenizer.decode(output_tokens)


def sample_top_p(probs, p):
  # probs sorted: [0.4499, 0.4071, ...]
  # probs cum_sum: [0.4499, 0.8571, ...]
  # safe probs_idx so we will remember the inital ordering of the vocabulary after mixing it up by sort
  probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
  probs_sum = torch.cumsum(probs_sort, dim=-1)
  # - probs_sort to shift by one, so we get masks for the first prob. that is greater than p
  mask = probs_sum - probs_sort > p
  probs_sort[mask] = 0.0
  # after removing many probability (masking them out) we have to redistribute the left over probabilities, so they sum up to 1 again
  probs_sort /= probs_sort.sum(dim=-1, keepdim=True)
  next_token = torch.multinomial(probs_sort, num_samples=1)
  # make use of index, by query using gather to retrieve the original token
  next_token = torch.gather(probs_idx, -1, next_token)
  return next_token.item()

# TODO: requirements.txt
# TODO: Fix bug generate printing
# TODO: Chris - add nice print statements + docstrings
# TODO: Chris - Tokenizer
# TODO: Chris - Checkout how quantization works
# TODO: Update README

# TODO: Thomas - GPU
# TODO: EVAL: runtime, attention, feedforward, layer time, first_token
# TODO: EVAL: inference_strategies (greedy vs. top-p)
# TODO: EVAL: benchmark MMLU: Augenmaß 32bit vs. 16bit vs. 8bit vs. 4bit
# TODO: EVAL: benchmark MMLU: LLaMA.cpp vs. own-implementation - das auch als Outlook nehmen

# TODO Bericht: Quantization erklärt + easy parallelization + by group to shrink error. wie sieht quantized forward pass aus, an welchen Stellen quant. vs. dequant., etc.

# TODO: implement chat-llama like function


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--bin", type=str, help="path to exported llama f32 weights", default= "bin/llama_q8.bin")
  parser.add_argument("--max-toks" , type=int, help="max tokens to generate", default=1000)
  parser.add_argument("prompt", type=str, nargs='*', help="prompt to generate from")

  args = parser.parse_args()
  full_prompt = ' '.join(args.prompt) if args.prompt else None

  rt = CreateRuntime(args.bin)

  generate_top_p(rt, full_prompt, max_toks=args.max_toks)
