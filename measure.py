# /// script
# dependencies = ["torch", "transformers==5.17.0", "peft==0.20.0"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
import json
import sys
from pathlib import Path

import torch

from prompt import DEFAULT_BASE, MAX_NEW_TOKENS, build_prompt, extract, load_causal_lm, load_tokenizer, system_prompt, user_message

IDS = ("asm00", "asm01", "asm02", "asm03")


def main():
    tok = load_tokenizer("./model", DEFAULT_BASE)
    model = load_causal_lm("./model", DEFAULT_BASE, dtype=torch.float32).to("cpu")
    if len(sys.argv) > 1:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sys.argv[1])
    model.eval()
    out = Path("answers")
    out.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in Path("bootcamp.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        if r["id"] not in IDS:
            continue
        prompt = build_prompt(system_prompt(), user_message(r["id"], r["statement"]))
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        answer = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        (out / f"{r['id']}.s").write_text(extract(answer))
        print(f"{r['id']} written", flush=True)


if __name__ == "__main__":
    main()
