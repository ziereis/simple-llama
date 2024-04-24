from qllama import CreateRuntime
from llama_utils import load_tokenizer
from qllama import Runtime, CreateRuntime
import os
import numpy as np
import torch

tokenizer = load_tokenizer("bin/tokenizer.model")

rt = CreateRuntime("bin/chat-llama_q4.bin", device="gpu")

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




sys_input = "you are an export python programmer and help me write python code"

sys_packed = f"[INST] <<SYS>> {sys_input} <<SYS>> [/INST]"

chat_input = "how to reverse a linked list in python?"

chat_packed = f"[INST] {chat_input} [/INST]"

print(generate_top_p(rt, chat_packed, 500, 0.3, 0.9))
