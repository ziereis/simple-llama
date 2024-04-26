import argparse
from qllama import  CreateRuntime
from generate import generate_text

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
  parser.add_argument("--bin", type=str, help="path to exported llama weights", default= "bin/llama_q8.bin")
  parser.add_argument("--max-toks", type=int, help="max tokens to generate", default=200)
  parser.add_argument("--method", type=str, help="generation method, available ones are \"greedy\", \"top_p\" and \"top_k\"", default="top_p")
  parser.add_argument("--temperature", type=float, help="temperature for generation", default=0.3)
  parser.add_argument("--top_p", type=float, help="top_p for generation", default=0.9)
  parser.add_argument("--top_k", type=int, help="top_k for generation", default=5)
  parser.add_argument("prompt", type=str, nargs='*', help="prompt to generate from")
  parser.add_argument('--gpu', action='store_true', help='use gpu for inference')

  args = parser.parse_args()
  device = "gpu" if args.gpu else "cpu"
  full_prompt = ' '.join(args.prompt) if args.prompt else None

  rt = CreateRuntime(args.bin, device=device)

  # call: echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null to make fair comparison
  generate_text(rt, full_prompt, args.max_toks, method=args.method, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
