from sentencepiece import SentencePieceProcessor
import argparse
import numpy as np
import os
from qllama import QLLama
from llama_utils import load_tokenizer

tokenizer = load_tokenizer("bin/tokenizer.model")

def generate_greedy(llama: QLLama, promt: str, max_toks: int = 30) -> str:
  input = tokenizer.encode(promt)
  output = []
  # feed the entire prompt as context
  for token in input:
    out = llama.forward(token, len(output))
    output.append(token)
    print(tokenizer.decode(output), end="\r")

  # generate the rest of the tokens
  for _ in range(max_toks - len(output)):
    out = llama.forward(output[-1], len(output))
    next_token = int(np.argmax(out))
    if next_token == tokenizer.eos_id():
      break
    output.append(next_token)
    print(tokenizer.decode(output), end="\r")
  print(tokenizer.decode(output))
  return tokenizer.decode(output)

# TODO: requirements.txt
# TODO: Fix bug generate printing
# TODO: Chris - add nice print statements + docstrings
# TODO: Chris - implement different generate function and evaluate quality
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
  parser.add_argument("--max-toks" , type=int, help="max tokens to generate", default=30)
  parser.add_argument("prompt", type=str, nargs='*', help="prompt to generate from")

  args = parser.parse_args()
  full_prompt = ' '.join(args.prompt) if args.prompt else None

  rt = QLLama(args.bin)
  generate_greedy(rt, full_prompt, max_toks=args.max_toks)
