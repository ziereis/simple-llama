from sentencepiece import SentencePieceProcessor
import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device('cpu')
import time
from pathlib import Path
import json
from model import ModelArgs, Transformer


def load_llama(checkpoints_dir: str, vocab_size: int, max_seq_len: int):
    prev_time = time.time()
    checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
    ckpt_path = checkpoints[0]
    print(f'Loading checkpoint "{ckpt_path}"')
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")

    with open(Path(checkpoints_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
        print(f"params: {params}")

    model_args = ModelArgs()
    model_args.max_seq_len = max_seq_len

    assert(model_args.dim == params['dim'])
    assert(model_args.n_layers == params['n_layers'])
    assert(model_args.vocab_size == vocab_size)
    assert(model_args.n_heads == params['n_heads'])
    assert(model_args.n_layers == params['n_layers'])

    model_args.vocab_size = vocab_size
    print(f"model_args: {model_args}")
    model = Transformer(model_args)

    del checkpoint['rope.freqs']
    model.load_state_dict(checkpoint, strict=True)
    print(f"Loaded model in {time.time() - prev_time:.2f}s")
    return model

def load_tokenizer(tokenizer_path: str):
    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer
