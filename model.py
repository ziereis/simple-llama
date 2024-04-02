import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device('cpu')
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


def compute_rotation(distance: int, head_dim: int):
  theta_numerator = torch.arange(0, head_dim, 2).float()
  theta = 1.0 / (10000 ** (theta_numerator / head_dim))
  val = distance * theta
  return torch.polar(torch.ones_like(val), val)

def rotate_embeddings(x: torch.Tensor, rotations: torch.Tensor):
    # (n_heads, head_dim) -> (n_heads, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (head_dim / 2) -> (1, head_dim / 2)
    # this is dont so it correctly broadcasts and applys to the same rotation
    # to every head
    rotations = rotations.unsqueeze(0)

    rotated = x_complex * rotations
    x_out = torch.view_as_real(rotated)
    x_out = x_out.reshape(*x.shape).type_as(x)
    return x_out

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    head_dim: int = 128
    hidden_dim: int = 11008
    vocab_size: int = 32000
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_seq_len: int = 2048


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_dim = args.head_dim
        self.n_heads = args.n_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_seq_len, args.n_heads, args.head_dim))
        self.cache_v = torch.zeros((args.max_seq_len, args.n_heads, args.head_dim))

    def forward(self, x: torch.Tensor, pos: int):
        #(embedding_dim)
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)


        # (n_heads, head_dim)
        xq = xq.view(self.n_heads, self.head_dim)
        xk = xk.view(self.n_heads, self.head_dim)
        xv = xv.view(self.n_heads, self.head_dim)


        # (head_dim / 2)
        rotations = compute_rotation(pos, self.head_dim)

        xq = rotate_embeddings(xq, rotations)
        xk = rotate_embeddings(xk, rotations)


        self.cache_k[pos] = xk
        self.cache_v[pos] = xv


        #retrieve all previous embeddings
        keys = self.cache_k[:pos+1]
        values = self.cache_v[:pos+1]


        # reshape so the head dim is the first dimension (can be ignored here doesnt matter)
        # so i basically just have a xq as vector of size head_dim
        # and keys and values as a matrix of size (len(0..pos), head_dim)
        xq = xq.unsqueeze(1)
        keys = keys.transpose(0,1)
        values = values.transpose(0,1)

        # now i want to compute the dot product of xq with all the embeddings
        # for all pervious positions

        # xq (1, head_dim)
        # keys (len(seq), head_dim)
        # keysTranspose (head_dim, len(seq))
        # xq matmul keysTranspose -> (1, len(seq))
        attention = torch.matmul(xq, keys.transpose(1,2)) / math.sqrt(self.head_dim)
        scores = F.softmax(attention, dim=-1)
        print(f"scores: {scores}")



        print(f"values: {values}")
        # scores (1, len(seq)) matmul (seq_len, head_dim) -> (1, head_dim)
        output = torch.matmul(scores, values)

        # (n_heads, 1, head_dim) -> (embedding_dim)
        output_joined = output.transpose(0, 1).contiguous().view(-1)
        print(f"out : {output_joined}")

        return self.wo(output_joined)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        assert x.shape == self.weight.shape
        return x * torch.rsqrt(x.pow(2).mean() + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        print(f"swish: {swish}")
        x_V = self.w3(x)
        print(f"x_V: {x_V}")
        #element wise
        x = swish * x_V
        x = self.w2(x)
        return x



class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.attention = SelfAttention(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.feed_forward = FeedForward(args)

    def forward(self, x: torch.Tensor, pos: int):
        normalized = self.attention_norm(x)
        att = self.attention(normalized, pos)
        print(f"attention: {att}")
        encoded = x + att
        print(f"resid: {encoded}")

        ffn_normalized = self.ffn_norm(encoded)
        print(f"normed: {ffn_normalized}")
        ffn_out = self.feed_forward(ffn_normalized)
        print(f"ffn: {ffn_out}")
        out = encoded + ffn_out
        return out



class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.params = args
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, token: int, pos: int):
        # take input token and query the embedding vector
        embedding = self.tok_embeddings(torch.tensor([token], dtype=torch.long))[0,:]
        print(f"embedding: {embedding}")
        for layer in self.layers:
            embedding = layer(embedding, pos)
        embedding_normalized = self.norm(embedding)
        output = self.output(embedding_normalized).float()
        return output



if __name__ == "__main__":
  model = Transformer(ModelArgs())


  with torch.no_grad():
    out = model(50, 5)

  print(out.shape)
