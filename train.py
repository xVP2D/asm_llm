#!/usr/bin/env python3
# /// script
# dependencies = ["torch", "transformers==5.17.0", "peft==0.20.0"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""LoRA fine-tuning of Qwen3-0.6B on a training jsonl. Writes a PEFT adapter directory
(adapter_config.json + adapter_model.safetensors), the format the grader loads.

Usage: train.py --tier <letter> --model <base-model> --out <adapter-out> [--epochs 2] [--rank 8] [--seed 1]
       train.py --model <base-model> --out <adapter-out> --data <train.jsonl> [...]
       uv run train.py --tier A --model ./model --out adapter --device cpu --epochs 1 --lr 5e-4 --rank 8 --seed 0
--tier <letter> trains on data/<letter>.jsonl unless --data overrides it.
<base-model> is a local directory or a Hugging Face id. GPU if one is visible, else CPU.
Each data line is either {"messages": [system, user, assistant]} (assistant content is the
complete ```nasm fenced answer) or the older {"id", "statement", "solution"} row.
--base olmo-3-7b-instruct: QLoRA (4-bit NF4, CUDA only), attention only, rank 16 at most."""
import argparse, json, random, sys, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from prompt import BASES, DEFAULT_BASE, QWEN_REVISION, SYSTEM, build_prompt, load_causal_lm, load_tokenizer, user_message

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="./model", help="base model: local directory or a Hugging Face id")
ap.add_argument("--out", default="adapter", help="directory the LoRA adapter is written to")
ap.add_argument("--tier", help="tier letter (A, B, ...): trains on data/<tier>.jsonl unless --data is given")
ap.add_argument("--data", help="training data jsonl; required unless --tier is given")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--rank", type=int, default=8, help="1 to 64: the grader refuses above 64")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--system", help="file holding your system prompt (the prompt.txt you commit); default: the grader's")
ap.add_argument("--template", default="server", choices=["server", "qwen-default"],
                help="server: the exact prompt the grading server builds (thinking block closed). "
                     "qwen-default: Qwen3's own chat template, for comparison only")
ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--base", default=DEFAULT_BASE, choices=sorted(BASES), help="the base <base-model> is a copy of")
ap.add_argument("--revision", help="Hugging Face revision; default: the one pinned for --base, when <base-model> is its id")
ap.add_argument("--metrics", help="also write timings and peak VRAM to this JSON file (bench.py)")
a = ap.parse_args()
if not a.data:
    if not a.tier:
        ap.error("--data or --tier is required")
    a.data = str(Path("data") / f"{a.tier}.jsonl")
base = BASES[a.base]
qlora = base["quant"] == "4bit"
if not 1 <= a.rank <= base["rank_max"]:
    ap.error(f"--rank must be between 1 and {base['rank_max']}")
if qlora and (a.template != "server" or a.device == "cpu"):
    ap.error(f"--base {a.base}: QLoRA needs CUDA and --template server")
# Qwen pinned to the snapshot the grading box caches: another snapshot can change the greedy answer.
revision = a.revision or (base["revision"] if a.model == base["model"] else None)
if revision is None and a.model == BASES[DEFAULT_BASE]["model"]:
    revision = QWEN_REVISION
rev = {"revision": revision} if revision else {}

device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
# bf16 only where the GPU has it (a T4 has not): elsewhere float32, slower but exact.
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
random.seed(a.seed); torch.manual_seed(a.seed)
system = Path(a.system).read_text().strip()[:2000] if a.system else SYSTEM

tok = load_tokenizer(a.model, a.base, **rev)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
rows = [json.loads(l) for l in open(a.data) if l.strip()]
samples = []
for i, r in enumerate(rows):
    if "messages" in r:
        msgs = r["messages"]
        row_system = next((m["content"] for m in msgs if m["role"] == "system"), system)
        user = next(m["content"] for m in msgs if m["role"] == "user")
        assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
    else:
        row_system = system
        user = user_message(r.get("id") or f"v{i:04d}", r["statement"])
        assistant = "```nasm\n" + r["solution"].strip() + "\n```"
    if a.template == "server":
        prompt = build_prompt(row_system, user, a.base)
    else:
        prompt = tok.apply_chat_template([{"role": "system", "content": row_system}, {"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(assistant + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids, labels = (p_ids + a_ids)[:a.max_len], ([-100] * len(p_ids) + a_ids)[:a.max_len]
    samples.append((torch.tensor(ids), torch.tensor(labels)))
print(f"{len(samples)} examples, device {device}, {str(dtype).split('.')[-1]}, rank {a.rank}, "
      f"{a.epochs} epochs, template {a.template}", flush=True)

if qlora:
    # As infra/gpu/trainer/train_qlora.py: NF4 + double quant, BF16 compute, no
    # prepare_model_for_kbit_training (it casts embeddings and lm_head to F32, +3.3 GB).
    from transformers import BitsAndBytesConfig
    if device != "cuda":
        sys.exit("QLoRA needs a CUDA device (bitsandbytes 4-bit)")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = load_causal_lm(a.model, a.base, quantization_config=bnb, dtype=torch.bfloat16,
                           device_map={"": 0}, **rev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
else:
    model = load_causal_lm(a.model, a.base, dtype=dtype, **rev).to(device)
# Plain LoRA only: the grading server merges W + (alpha/r)·B·A and refuses DoRA,
# rsLoRA, a bias and modules_to_save.
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=list(base["targets"])))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = a.epochs * len(samples)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
model.train(); t0 = time.time()
for ep in range(a.epochs):
    tot = 0.0
    for i in torch.randperm(len(samples)).tolist():
        ids, labels = samples[i]
        out = model(input_ids=ids[None].to(device), labels=labels[None].to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(); tot += out.loss.item()
    print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(samples):.3f} {time.time() - t0:.0f}s", flush=True)
model.save_pretrained(a.out)
# adapter/ holds the two PEFT files only (#2114): the grader refuses the README save_pretrained writes.
(Path(a.out) / "README.md").unlink(missing_ok=True)
seconds = time.time() - t0
meta = {"data": a.data, "examples": len(samples), "epochs": a.epochs, "rank": a.rank, "lr": a.lr, "seed": a.seed,
        "template": a.template, "device": device, "dtype": str(dtype), "seconds": round(seconds)}
if a.base != DEFAULT_BASE:
    meta.update(base=a.base, quant=base["quant"], targets=list(base["targets"]), **rev)
size = (Path(a.out) / "adapter_model.safetensors").stat().st_size
if a.metrics:
    cuda = device == "cuda"
    json.dump({**meta, "train_seconds": round(seconds, 2), "train_tokens": sum(len(s[0]) for s in samples) * a.epochs,
               "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() // 2**20 if cuda else None,
               "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() // 2**20 if cuda else None,
               "adapter_bytes": size}, open(a.metrics, "w"), indent=1)
print(f"adapter written to {a.out}/ ({size / 2**20:.1f} MiB) in {seconds:.0f} s")
