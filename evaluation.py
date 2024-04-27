# TODO
# create some looping code here to:
# - comparison cpu vs gpu
# - run 4bit, 8bit, 32bit
# - compare llama.cpp vs. own implementation
# - compare token/s, firsttoken_duration, avg_token_time
# - compare output mit augenmaß
# - compare different inference strats

# TODO: requirements.txt
# TODO: Fix bug generate printing
# TODO: Chris - add nice print statements + docstrings - ⁠zeitmessen: in matrix muls, etc. (Pfeile in Graphik)
# TODO: Chris - Tokenizer
# TODO: Chris - Checkout how quantization works
## Add in report

# TODO: Update README
# TODO: EVAL: runtime, attention, feedforward, layer time, first_token
# TODO: EVAL: inference_strategies (greedy vs. top-p)
# TODO: EVAL: benchmark MMLU: Augenmaß 32bit vs. 16bit vs. 8bit vs. 4bit
# TODO: EVAL: benchmark MMLU: LLaMA.cpp vs. own-implementation - das auch als Outlook nehmen

## Results
# - inference strategies
# - ⁠quantization: speed vs. Quality
#     - ⁠quantization error
#     - benchmark MMLU: LLaMA.cpp vs. own-implementation
#     - 32bit, 8bit, 4bit
#     - tabelle mit example outputs
#     - ⁠metrik: tokens/sec, first_token
#     - ⁠zeitmessen: in matrix muls, etc. (Pfeile in Graphik)

## Ausblick
# - Bessere Quantization teqniques
# - Weitere Inference Strategies e.g. Beamsearch