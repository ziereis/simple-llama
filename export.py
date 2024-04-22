from llama_utils import load_llama, load_tokenizer
import argparse
import struct
import torch
from tqdm import tqdm
from model import Transformer


def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize(filename: str, model: Transformer):
  version = 3
  magic = 0x7fdd7f7f
  out_file = open(filename, 'wb')
  out_file.write(struct.pack('I', magic))
  out_file.write(struct.pack('i', version))
  p = model.params
  header = struct.pack('iiiiii', p.dim, p.hidden_dim, p.n_heads, p.n_layers, p.vocab_size, p.max_seq_len)
  out_file.write(header)
  pad = 256 - out_file.tell()
  assert pad >= 0
  out_file.write(b'\0' * pad)
  weights = [
        model.norm.weight,
        model.tok_embeddings.weight,
        model.output.weight,
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
  for w in tqdm(weights, desc="serializing"):
    serialize_fp32(out_file, w)
  out_file.close()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--default-dir", type=str, help="path to default model dir", default="bin/llama-2-7b")
  parser.add_argument("--tok-path", type=str, help="path to tokenizer model", default="bin/tokenizer.model")
  parser.add_argument("--max_seq_len", type=int, help="max sequence length", default=1024)
  parser.add_argument("--bin-dir", type=str, help="path to bin directory", default="bin")

  args = parser.parse_args()

  load_tokenizerk = load_tokenizer(args.tok_path)
  print(f"loading model from {args.default_dir} ")
  model = load_llama(args.default_dir, load_tokenizerk.vocab_size(), args.max_seq_len)
  print(f"serializing model to {args.bin_dir}/llama.bin")
  serialize(f"{args.bin_dir}/chat-llama.bin", model)
