# LLaMA2 Inference and Quantization

Welcome to the LLaMA2 Inference and Quantization project! This repository provides tools and libraries for running and
optimizing the LLaMA2 7B model for text generation. This guide will help you set up the environment, convert the model
weights, and perform efficient inference with quantization support.

## Requirements

### Mandatory

- Linux (Mac might also work)
- Python 3
- pip
- A C compiler (tested on gcc and clang)

### Optional

- CUDA (nvcc) for GPU support

## Setup Guide

### 1. Install Dependencies

Run the following command to install all required Python packages:

```sh
pip install -r requirements.txt
```

### 2. Download Model Weights
Download the Meta LLaMA2 7B weights from [here](https://llama.meta.com/llama-downloads). **We highly recommend downloading the llama2-7b-chat model** for superior text generation performance.

### 3. Organize Model Files
Create a `/bin` directory in the project root and move the `llama-2-7b-chat` directory and `tokenizer.model` file into the /bin directory.

### 4. Export Model Weights
Export the model weights into our binary file format by running:
```sh
python3 export.py --help
```
*Note: This process requires at least ~28GB of peak RAM.*

### 5. Compile the C/CUDA Library
Compile the library used for quantization and model inference with the following commands:
```sh
mkdir build && cd build
cmake .. -DCUDA=ON # with CUDA support
cmake .. # without CUDA
make
```

### 6. Run Text Completion
You can now run text completion using the base model:
```sh
python3 run.py --bin=bin/chat-llama.bin "Richard Feynman was a "
```
*Note: This can be slow. For better performance, quantize the model weights as described below.*


### Quantization for Improved Performance
We support 8-bit and 4-bit quantization to speed up inference. For GPUs, only a 4-bit forward pass is implemented to accommodate hardware constraints, allowing it to run on almost any GPU with around 6GB of VRAM.

#### Quantize the Model Weights
Run the quantization script:
```sh
python3 quantize.py --help
```

### Inference Tips
* Adjust text generation parameters like type (top_p, top_k, greedy) and temperature to improve inference quality.
* For prompt engineering, refer to the `chat.py` file.
* For a deeper understanding of LLaMA2, check out `model.py` for a simple PyTorch implementation, and explore the CUDA/C version in `/lib`.

# LLama2 model information

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

### Outputs

* norm (embedding_dim) = (4096)
* output (embedding_dim, vocab_size) = (4096, 32000)

![LLaMA2 Architecture](llama_architecture.png)

### Understanding the Code
#### Python Files
* **generate.py**: Contains the main function for text generation using different sampling methods (greedy, top_p, top_k).
* **qllama.py**: Defines the `Runtime` classes for various quantization levels (F32, Q8, Q4) and includes GPU support.
* **llama_utils.py**: Provides utility functions to load the model and tokenizer.
* **quantize.py**: Implements the quantization process for the model weights.
* **run.py**: Script to run the text generation using the quantized or non-quantized models.

#### C/CUDA Files
* **llama.c**: Core implementation for initializing and running the LLaMA model, supporting both full precision and quantized versions.
* **ops.c**: Contains various mathematical operations used in model inference and quantization.
* **quantize.c**: Handles the quantization process, converting model weights to 8-bit or 4-bit precision.
* **cu-llama.c**: CUDA-specific implementations to run the model on GPU, including memory management and kernel launches.
* **cu-ops.c**: CUDA operations for quantization, dequantization, matrix-vector multiplications, and other neural network operations.


# References:

* https://github.com/karpathy/llama2.c
* https://youtu.be/kCc8FmEb1nY
* https://youtu.be/oM4VmoabDAI

We hope you find this guide helpful! Happy experimenting with LLaMA2!

