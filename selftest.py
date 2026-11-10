#!/usr/bin/env python3
"""Local run of the asm_llm basic tests on the 11 public statements, before you push.

Same prompt as the grading server (see prompt.py), greedy decoding, 700 new tokens,
then each program is assembled with nasm/ld and replayed against its cases. The three
basic tests of the platform are reproduced: adapter, build public, gate (6 of 11).

Usage: selftest.py <base-model> [--adapter adapter] [--repo .] [--statements public.jsonl]
       selftest.py Qwen/Qwen3-0.6B --adapter adapter
Needs nasm and ld on PATH. Exit 0 when the three basic tests would pass.
--adapter none measures the bare model: no gate, exit 0, score and generation time printed.

Not identical to the server: it runs the adapter through PEFT in PyTorch, the server
merges it into the weights in its own engine. Expect the same programs on most
statements, not on all of them (see the README for the measured agreement)."""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompt import (BASES, DEFAULT_BASE, MAX_NEW_TOKENS, MINISTRAL, build_prompt, extract, load_causal_lm,  # noqa: E402
                    load_tokenizer, system_prompt, user_message)  # noqa: E402

ADAPTER_MAX = 128 * 1024 * 1024
RANK_MAX = 64
GATE_MIN = 6


def check_adapter(d, base=DEFAULT_BASE):
    """grade.py adapter, plus the refusals of the inference server."""
    rank_max = BASES[base]["rank_max"]
    cfg, w = d / "adapter_config.json", d / "adapter_model.safetensors"
    if not cfg.is_file() or not w.is_file():
        return "adapter/ must hold adapter_config.json and adapter_model.safetensors"
    if w.stat().st_size > ADAPTER_MAX:
        return f"adapter_model.safetensors is {w.stat().st_size // 2**20} MiB, the cap is 128 MiB"
    if cfg.stat().st_size > 1024 * 1024:
        return "adapter_config.json is above 1 MiB"
    try:
        c = json.loads(cfg.read_text(errors="replace"))
    except ValueError:
        return "adapter_config.json is not valid JSON"
    r = c.get("r")
    if not isinstance(r, int) or isinstance(r, bool) or not 1 <= r <= rank_max:
        return f"r must be an integer between 1 and {rank_max} (got {r!r})"
    if c.get("peft_type") != "LORA":
        return "peft_type must be LORA"
    for flag in ("use_dora", "use_rslora"):
        if c.get(flag) is True:
            return f"{flag} is true: the server only merges plain LoRA"
    if c.get("bias") not in (None, "none"):
        return "bias must be none"
    if c.get("modules_to_save"):
        return "modules_to_save must be empty"
    if not isinstance(c.get("target_modules"), list) or not c["target_modules"]:
        return "target_modules must be a non-empty list of module names"
    if "lora_alpha" not in c:
        return "lora_alpha is missing"
    if base != DEFAULT_BASE and not set(c["target_modules"]) <= set(BASES[base]["targets"]):
        return f"target_modules must be among {', '.join(BASES[base]['targets'])} for {base}"
    return f"adapter ok, r={r}, {w.stat().st_size // 2**20} MiB"


def run_case(d, t):
    try:
        p = subprocess.run(["bash", "-c", "ulimit -f 1024; ulimit -v 262144; " + t["command"]], cwd=d,
                           capture_output=True, text=True, errors="replace", timeout=5, stdin=subprocess.DEVNULL)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return "timeout", ""


def grade(row, src, work):
    """(built, passed, explanation) with the messages of grade.py for a public statement."""
    d = work / row["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "prog.s").write_text(src)
    b = subprocess.run(["sh", "-c", "nasm -f elf64 prog.s -o prog.o && ld prog.o -o prog"], cwd=d,
                       capture_output=True, text=True, errors="replace", timeout=20)
    if b.returncode:
        first = b.stderr.strip().splitlines()[0][:160] if b.stderr.strip() else ""
        return False, False, "does not assemble" + (f": {first}" if first else "")
    for t in row["tests"]:
        code, out = run_case(d, t)
        if code != t["exit"] or (t["stdout"] is not None and out != t["stdout"]):
            return True, False, (f"command:  {t['command']}\n  expected: exit {t['exit']}, stdout {t['stdout']!r}\n"
                                 f"  got:      exit {code}, stdout {out[:200]!r}")
    return True, True, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Qwen3-0.6B: local directory or Hugging Face id")
    ap.add_argument("--adapter", default="adapter", help="adapter directory; 'none' measures the bare model, no gate")
    ap.add_argument("--repo", default=".", help="where prompt.txt is read from, as on the runner")
    ap.add_argument("--statements", default=str(HERE / "public.jsonl"))
    ap.add_argument("--out", help="keep the generated programs here (default: a temporary directory)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--merge", action="store_true", help="merge the adapter into the weights before generating")
    ap.add_argument("--base", default=DEFAULT_BASE, choices=sorted(BASES), help="the base <model> is a copy of")
    ap.add_argument("--revision", help="Hugging Face revision; default: the one pinned for --base, when <model> is its id")
    ap.add_argument("--metrics", help="also write tokens/s and peak VRAM to this JSON file (bench.py)")
    a = ap.parse_args()
    base = BASES[a.base]
    revision = a.revision or (base["revision"] if a.model == base["model"] else None)
    rev = {"revision": revision} if revision else {}
    for tool in ("nasm", "ld"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is not on PATH (apt install nasm binutils)")

    import torch

    ok_adapter = True
    if a.adapter != "none":
        msg = check_adapter(Path(a.adapter), a.base)
        ok_adapter = msg.startswith("adapter ok")
        print(("PASS" if ok_adapter else "FAIL"), "adapter:", msg)
        if not ok_adapter:
            sys.exit(1)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    # OLMo too is generated from the BF16 base, as the Engine merges a QLoRA adapter into it.
    tok = load_tokenizer(a.model, a.base, **rev)
    model = load_causal_lm(a.model, a.base, dtype=dtype, **rev).to(device)
    if a.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        if a.merge:
            model = model.merge_and_unload()
    model.eval()
    stop = [i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} if i is not None]
    if a.base in MINISTRAL:   # no <|im_end|> in this vocabulary: it maps to <unk>
        stop = [tok.eos_token_id]

    system = system_prompt(a.repo)
    rows = [json.loads(l) for l in Path(a.statements).read_text().splitlines() if l.strip()]
    work = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="selftest-"))
    built = passed = 0
    report, t0, gen_s = [], time.time(), 0.0
    for r in rows:
        prompt = build_prompt(system, user_message(r["id"], r["statement"]), a.base)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        tg = time.time()
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=stop,
                               pad_token_id=tok.eos_token_id)
        new = g[0][ids["input_ids"].shape[1]:]
        gen_s += time.time() - tg
        text = tok.decode(new, skip_special_tokens=True)
        src = extract(text)
        b, p, why = grade(r, src, work)
        built += b; passed += p
        (work / r["id"] / "answer.txt").write_text(text)
        report.append({"id": r["id"], "family": r["family"], "built": b, "pass": p, "tokens": len(new)})
        print(f"{r['id']} {r['family']:<10} {'ok' if p else 'ko'}" + ("" if p else f"\n  statement: {r['statement']}\n  {why}"),
              flush=True)
    n = len(rows)
    seconds = time.time() - t0
    # As on the platform, "build public" passes whatever the count: a program that
    # does not assemble fails at the gate, not here.
    print(f"\nPASS build public: {built}/{n} programs assembled")
    if a.adapter == "none":
        # No adapter, no gate: the bare model's score is the measure to beat, exit 0 (#1945).
        threads = f", {torch.get_num_threads()} threads" if device == "cpu" else ""
        print(f"baseline, no gate: {passed}/{n} public statements pass before training; "
              f"generation {seconds:.0f} s on {device}{threads}")
        print(f"programs in {work}")
    else:
        print(f"{'PASS' if passed >= GATE_MIN else 'FAIL'} gate: {passed}/{n} pass, {GATE_MIN} needed to be graded")
        print(f"generation {seconds:.0f}s on {device}; programs in {work}")
    (work / "selftest.json").write_text(json.dumps({"built": built, "passed": passed, "n": n, "rows": report}, indent=1))
    if a.metrics:
        tokens = sum(x["tokens"] for x in report)
        cuda = device == "cuda"
        Path(a.metrics).write_text(json.dumps({
            "base": a.base, **rev, "device": device, "built": built, "passed": passed, "n": n,
            "gen_seconds": round(gen_s, 2), "gen_tokens": tokens, "tokens_per_s": round(tokens / max(gen_s, 1e-9), 2),
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() // 2**20 if cuda else None,
            "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() // 2**20 if cuda else None}, indent=1))
    sys.exit(0 if ok_adapter and (a.adapter == "none" or passed >= GATE_MIN) else 1)


if __name__ == "__main__":
    main()
