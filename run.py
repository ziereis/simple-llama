import argparse
from qllama import  CreateRuntime
from generate import generate_text


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--bin", type=str, help="path to exported llama f32 weights", default= "bin/llama_q8.bin")
  parser.add_argument("--max-toks", type=int, help="max tokens to generate", default=200)
  parser.add_argument("prompt", type=str, nargs='*', help="prompt to generate from")

  args = parser.parse_args()
  full_prompt = ' '.join(args.prompt) if args.prompt else None

  rt = CreateRuntime(args.bin)

  sys_input = "you are an export python programmer and help me write python code"

  sys_packed = f"[INST] <<SYS>> {sys_input} <<SYS>> [/INST]"

  chat_input = "how to reverse a linked list in python?"

  chat_packed = f"[INST] {full_prompt} [/INST]"

  # call: echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null to make fair comparison
  generate_text(rt, chat_packed, args.max_toks, method='top_p', temperature=0.3, top_p=0.9)
