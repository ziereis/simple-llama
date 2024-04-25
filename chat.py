from qllama import CreateRuntime
from generate import generate_text



rt = CreateRuntime("bin/chat-llama_q4.bin", device="gpu")
sys_input = "you are an export python programmer and help me write python code"

sys_packed = f"[INST] <<SYS>> {sys_input} <<SYS>> [/INST]"

chat_input = "what kind of programming language is ocaml? tell if its compiled to machine code or interpreted and what makes is unqiue"

chat_packed = f"[INST] {chat_input} [/INST]"

generate_text(rt, chat_packed, 100, method="top_p", temperature=0.3, top_p=0.8)
