# /// script
# dependencies = ["torch", "transformers==5.17.0", "peft==0.20.0"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Asks Qwen3-0.6B + your adapter the statement of asm00, as the runner replays it:
kit prompt, CPU, float32, greedy, 200 new tokens. Writes asm00.s, builds it, runs it.
Usage: uv run try.py adapter"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

for tool in ("nasm", "ld"):
    if not shutil.which(tool):
        sys.exit(f"{tool} is not on PATH (apt install nasm binutils)")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import QWEN_REVISION, SYSTEM, build_prompt, extract, user_message

# ./model is where get_model.py puts the pinned snapshot (v2, 2026-11-08); the Hub id is the fallback.
MODEL = "./model" if Path("model/MODEL.json").is_file() else "Qwen/Qwen3-0.6B"
STATEMENT = "Create a program that exits with code 0."
MAX_NEW_TOKENS = 200

adapter = sys.argv[1] if len(sys.argv) > 1 else "adapter"
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=QWEN_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=QWEN_REVISION, dtype=torch.float32)
model = PeftModel.from_pretrained(model, adapter).eval()
config = model.peft_config["default"]

prompt = build_prompt(SYSTEM, user_message("asm00", STATEMENT))
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
program = extract(answer)
Path("asm00.s").write_text(program)

build = subprocess.run("nasm -f elf64 asm00.s -o asm00.o && ld asm00.o -o asm00",
                       shell=True, capture_output=True, text=True)
code, stdout = None, None
if build.returncode == 0:
    try:
        run = subprocess.run(["./asm00"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
        code = run.returncode if run.returncode >= 0 else 128 - run.returncode
        stdout = run.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        code, stdout = 124, ""

print("---INFO---")
print(f"MODEL          = {MODEL}")
print(f"adapter        = {adapter} (r={config.r}, lora_alpha={config.lora_alpha})")
print(f"device         = {model.device}")
print(f"dtype          = {model.dtype}")
print(f"max_new_tokens = {MAX_NEW_TOKENS}")
print("do_sample      = False")
print("---PROGRAM---")
print(program, end="")
print("---RESULT---")
print(f"builds = {str(build.returncode == 0).lower()}")
print(f"exit   = {json.dumps(code)}")
print(f"stdout = {json.dumps(stdout)}")
