
* create a bin folder in the root dir
* copy your meta llama-2-7b folder and tokenizer.model file there
* export.py to convert the meta llama model to a fp32 base version
* quantize.py to convert fp32 version to int8 quantized version
* run.py run inference (sentence completion)


### state dict contains the following weights:

#### Input
* tok_embeddings (vocab_size, embedding_dim) = (32000,4096)
### Layer 0-31
#### Attention
* attention_norm (embedding_dim) = (4096)
* attention.wq (embedding_dim, embedding_dim) = (4096, 4096)
* attention.wk (embedding_dim, embedding_dim) = (4096, 4096)
* attention.wv (embedding_dim, embedding_dim) = (4096, 4096)
* attention.w0 (embedding_dim, embedding_dim) = (4096, 4096)
#### FeedFordward
* feed_forward.norm (embedding_dim) = (4096)
* feed_forward.w1 (embedding_dim, hidden_dim) = (4096, 11008)
* feed_forward.w3 (embedding_dim, hidden_dim) = (4096, 11008)

(w1 and w3 get both applied to the input embeddings and then element wise multiplied)
* feed_forward.w2 (hidden_dim, embedding_dim) = (11008, 4096)



## Output
* norm (embedding_dim) = (4096)
* output (embedding_dim, vocab_size) = (4096, 32000)

References:
* https://github.com/karpathy/llama2.c
* https://youtu.be/kCc8FmEb1nY
* https://youtu.be/oM4VmoabDAI
