"""The prompt exactly as the grading pipeline builds it. Train and self-test on this,
or the model you train is not the model that gets graded.

Mirrors, at the time of writing (2026-09-25):
  - the runner: system = prompt.txt (stripped, 2000 chars max) or SYSTEM below,
    user = "Exercise <id>. File: <id>.s\\n\\n<statement>", max 700 new tokens;
  - the inference server: Qwen3 chat markup with an EMPTY thinking block already
    closed, greedy decoding, stop on <|im_end|>.
"""
import re
from pathlib import Path

SYSTEM = ("You write x86-64 assembly for Linux in NASM syntax, without libc, using raw "
          "syscalls (exit = 60, write = 1, read = 0). Answer with one ```nasm code block "
          "containing the complete program and nothing else.")
MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS = 700
_CODE_RE = re.compile(r"```(?:nasm|asm|assembly|x86asm)?\s*\n(.*?)```", re.S)


def system_prompt(repo="."):
    """prompt.txt at the repository root, as the runner reads it; missing or empty = SYSTEM."""
    p = Path(repo) / "prompt.txt"
    if not p.is_file():
        return SYSTEM
    text = p.read_bytes()[:MAX_PROMPT_CHARS * 4].decode("utf-8", "replace").strip()
    return text[:MAX_PROMPT_CHARS] if text else SYSTEM


def user_message(vid, statement):
    return f"Exercise {vid}. File: {vid}.s\n\n{statement}"


def build_prompt(system, user, base="qwen3-0.6b"):
    if base == "olmo-3-7b-instruct":
        # Olmo-3-7B-Instruct chat_template.jinja at OLMO_REVISION, add_generation_prompt=True
        # (fixture: tests/fixtures/olmo3_chat.json). No thinking block.
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if base in MINISTRAL:
        # Ministral-3-*-Instruct-2512 chat_template.jinja, add_generation_prompt=True, explicit system
        # (fixture: tests/fixtures/ministral3_chat.json). BOS is part of the text: tokenize without specials.
        return f"<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST]"
    if base != "qwen3-0.6b":
        raise ValueError(f"unknown base {base!r}")
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def extract(text):
    """First ```nasm block (or the whole answer if there is none), as the runner does."""
    m = _CODE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip() + "\n"


# Base models the kit can train. Mirrors infra/gpu/common/x0ne_gpu_common/bases.py (rank_max,
# quant, targets); checked by tests/test_kit.py. OLMo 3 7B: attention only, rank <= 16, 4-bit.
DEFAULT_BASE = "qwen3-0.6b"
OLMO_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
# The Qwen3-0.6B snapshot the runner image caches (llm-image role, runner_llm_model_revision).
# train.py and try.py pin it so the replayed generation matches; BASES keeps revision None.
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BASES = {
    DEFAULT_BASE: {"model": "Qwen/Qwen3-0.6B", "revision": None, "quant": None, "rank_max": 64,
                   "targets": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")},
    "olmo-3-7b-instruct": {"model": "allenai/Olmo-3-7B-Instruct", "revision": OLMO_REVISION, "quant": "4bit",
                           "rank_max": 16, "targets": ("q_proj", "k_proj", "v_proj", "o_proj")},
}
# Bench-only bases (TASKS #1823), not in bases.py: no platform worker or Engine serves them.
# Mistral3ForConditionalGeneration checkpoints; the kit loads the text model only (load_causal_lm).
MINISTRAL = {
    "ministral-3-8b-instruct": {"model": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
                                "revision": "f6fae9795746f63c9be8344932f01275f3c63734", "quant": "4bit",
                                "rank_max": 16, "targets": ("q_proj", "k_proj", "v_proj", "o_proj")},
    "ministral-3-3b-instruct": {"model": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
                                "revision": "b6d637bef2393152b3da2b2fde72eecdee30557e", "quant": None, "rank_max": 64,
                                "targets": BASES[DEFAULT_BASE]["targets"]},
}
PLATFORM_BASES = tuple(BASES)
BASES.update({k: {**v, "platform": False} for k, v in MINISTRAL.items()})
# Checkpoint keys are language_model.model.* / language_model.lm_head.*; the vision tower is left unloaded.
_MINISTRAL_KEYS = {r"^language_model\.model\.": "model.", r"^language_model\.lm_head\.": "lm_head."}


def load_tokenizer(path, base, **kw):
    from transformers import AutoTokenizer
    if base in MINISTRAL:
        kw["fix_mistral_regex"] = True   # transformers 5.17 warns the shipped pre-tokenizer regex is wrong
    return AutoTokenizer.from_pretrained(path, **kw)


def load_causal_lm(path, base, **kw):
    """AutoModelForCausalLM, or for Ministral the text model of the Mistral3 checkpoint."""
    if base not in MINISTRAL:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(path, **kw)
    from transformers import AutoConfig, Ministral3ForCausalLM
    cfg = AutoConfig.from_pretrained(path, revision=kw.get("revision"))
    return Ministral3ForCausalLM.from_pretrained(path, config=cfg.text_config, key_mapping=_MINISTRAL_KEYS, **kw)

