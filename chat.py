from qllama import CreateRuntime
from generate import generate_text
import time


start = time.time()
rt = CreateRuntime("bin/chat-llama_q4.bin")
end = time.time()

sys_input = "give a short and concise answer"

sys_packed = f"[INST] <<SYS>> {sys_input} <<SYS>> [/INST]"

chat_input = "Richard Feynman was a "

chat_packed = f"""[INST] Given two Boolean random variables, A and B, where P(A) = 1/2, P(B) = 1/3, and P(A | ¬B) = 1/4, what is P(A | B)? pick among the following:
[ "1/6", "1/4", "3/4", "1" ] [/INST]"""
  # call: echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null to make fair comparison
generate_text(rt, sys_packed + chat_packed, 1000, method="greedy", temperature=0.3, top_p=0.8)
print("Runtime creation time: ", end - start)
