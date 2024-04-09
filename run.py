from sentencepiece import SentencePieceProcessor
import argparse
import subprocess

def load_tokenizer(tokenizer_path: str):
    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("executeable", type=str, help="path to the llama executable", default= "build/llama")
  parser.add_argument("weights", type=str, help="path to the weights file", default = "llama.bin")
  parser.add_argument("input", metavar="N", type=str, nargs="+", help="input tokens")


  args = parser.parse_args()

  tokenizer = load_tokenizer("bin/tokenizer.model")
  encoded = tokenizer.encode(" ".join(args.input))
  encoded = [str(x) for x in encoded]
  out_args = [args.executeable, args.weights] + encoded

  result = subprocess.run(out_args, capture_output=True, text=True)
  print(result.stderr)

  tokens = result.stdout.split(" ")
  tokens = [int(x) for x in tokens if x != ""]
  print(tokenizer.decode(tokens))
