#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub"]
# ///
"""Puts the pinned Qwen3-0.6B snapshot in ./model, the directory every kit script and the
grader load. Plain files, no symlinks; model/ is ignored by git (see .gitignore), the grader
mounts its own copy of the same snapshot at the same path and refuses committed weights.

Usage: get_model.py [model-dir]        default: ./model
Needs the network once; no Hugging Face account (the repository is public, not gated)."""
import hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompt import BASES, DEFAULT_BASE, QWEN_REVISION  # noqa: E402

FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "model")
    if (out / "MODEL.json").is_file():
        sys.exit(f"{out}/MODEL.json exists: the model is already there")
    from huggingface_hub import constants, snapshot_download
    repo = BASES[DEFAULT_BASE]["model"]
    # Cached snapshot first: offline, snapshot_download still asks the Hub for the tree
    # (huggingface_hub 1.x, measured 2026-11-08) and fails where the cache would do.
    snap = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / QWEN_REVISION
    if not all((snap / f).exists() for f in FILES):
        snap = Path(snapshot_download(repo, revision=QWEN_REVISION))
    out.mkdir(parents=True, exist_ok=True)
    sums = {}
    for name in FILES:
        src = snap / name
        if not src.exists():
            sys.exit(f"{name} missing from the snapshot {QWEN_REVISION[:8]}")
        shutil.copyfile(src, out / name)          # resolves the cache symlink: a plain file
        sums[name] = sha256(out / name)
    (out / "MODEL.json").write_text(json.dumps({"model": repo, "revision": QWEN_REVISION, "sha256": sums}, indent=1) + "\n")
    size = sum((out / n).stat().st_size for n in FILES)
    print(f"{repo} @ {QWEN_REVISION[:8]} in {out}/ ({size / 2**30:.2f} GiB), never commit it")


if __name__ == "__main__":
    main()
