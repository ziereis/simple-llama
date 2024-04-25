import torch
from sentencepiece import SentencePieceProcessor
import argparse
import numpy as np
import os
from qllama import Runtime, CreateRuntime
from llama_utils import load_tokenizer
import time

tokenizer = load_tokenizer("bin/tokenizer.model")


def generate_text(llama: Runtime, prompt: str, max_toks: int = 30, method: str = 'greedy', temperature: float = 0.1,
                  top_p: float = 0.1, top_k: int = 10000) -> str:
  """
     Generates text based on a given prompt using a specified sampling method.

     Args:
         llama (Runtime): The LLaMA model runtime instance.
         prompt (str): The initial text to seed the generation.
         max_toks (int): Maximum number of tokens to generate.
         method (str): The method of sampling to use ('greedy', 'top_p', 'top_k').
         temperature (float): Temperature scaling factor for probability distribution. Lower values lead to less randomness.
         top_p (float): Cumulative probability threshold for top-p sampling (used only if method='top_p').
         top_k (int): Number of top tokens considered for sampling (used only if method='top_k').

     Returns:
         str: The generated text as a string.
     """

  assert method in ['greedy', 'top_p', 'top_k'], "Invalid method specified. Use 'greedy', 'top_p', or 'top_k'."

  input_tokens = tokenizer.encode(prompt)
  output_tokens = []


  start_time = time.time()
  first_token = input_tokens[0]
  _ = llama.forward(first_token, len(output_tokens))
  output_tokens.append(first_token)
  first_token_time = time.time() - start_time

  # Feed in Prompt to fill KV Cache
  for token in input_tokens[1:]:
    _ = llama.forward(token, len(output_tokens))
    output_tokens.append(token)
    os.system('clear')
    print(tokenizer.decode(output_tokens))

  # Completion
  while len(output_tokens) < max_toks:
    latest_token = output_tokens[-1]
    out = llama.forward(latest_token, len(output_tokens))
    out = torch.tensor(out)

    if method == 'greedy':
      next_token = torch.argmax(out).item()
    elif method == 'top_p':
      next_token = sample_top_p(torch.softmax(out / temperature, dim=-1), top_p)
    elif method == 'top_k':
      next_token = sample_top_k(torch.softmax(out / temperature, dim=-1), top_k)

    if next_token == tokenizer.eos_id():
      break
    output_tokens.append(next_token)
    os.system('clear')
    print(tokenizer.decode(output_tokens))
  os.system('clear')
  print(tokenizer.decode(output_tokens))

  # Calculate tokens per second
  total_time = time.time() - start_time
  average_token_time = total_time / len(output_tokens)
  tokens_per_second = len(output_tokens) / total_time

  # Output performance metrics
  print(f"Time to generate first token: {first_token_time:.2f} seconds")
  print(f"Average time per token: {average_token_time:.2f} seconds")
  print(f"Tokens per second: {tokens_per_second:.2f}")

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

def sample_top_k(probs, k):
  top_k_probs, top_k_indices = torch.topk(probs, k)
  next_token = top_k_indices[torch.multinomial(top_k_probs, num_samples=1)]
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
  parser.add_argument("--max-toks", type=int, help="max tokens to generate", default=40)
  parser.add_argument("prompt", type=str, nargs='*', help="prompt to generate from")

  args = parser.parse_args()
  full_prompt = ' '.join(args.prompt) if args.prompt else None

  rt = CreateRuntime(args.bin)

  sys_input = "you are an export python programmer and help me write python code"

  sys_packed = f"[INST] <<SYS>> {sys_input} <<SYS>> [/INST]"

  chat_input = "how to reverse a linked list in python?"

  chat_packed = f"[INST] {full_prompt} [/INST]"

  generate_text(rt, chat_packed, max_toks=args.max_toks, method="top_k")
  # call: echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null to make fair comparison
