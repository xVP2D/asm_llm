# asm_llm, starter kit (minimal)

Twelve files. `get_model.py` puts Qwen3-0.6B in `./model`; `bootcamp.jsonl` holds the statements your
`measure.py` asks it (step 1); `train.py` trains a LoRA adapter on your
CPU and `selftest.py` replays the public statements locally (steps 2 to 4). The full kit (data
generator, bench, Colab notebook) opens in the activity once step 1 is green.

| File | What it is |
|---|---|
| `get_model.py` | Copies the pinned Qwen3-0.6B snapshot into `./model/` (1.5 GB, plain files) and writes `model/MODEL.json`. Network once; no account. |
| `.gitignore` | Keeps `model/`, every weight file but `adapter/adapter_model.safetensors`, `.venv/` and `*.o` out of git. A committed weight fails every step. |
| `proof/before.json` | The bare model on the 23 bootcamp statements, asm00 to asm22, each program run against the bootcamp tests, no adapter. For reference: your `measure.py` makes your own before. |
| `train.py` | LoRA training, plain PEFT: rank 1 to 64, alpha = 2·rank, the 7 projections. Writes `adapter/`. |
| `try.py` | Asks the model with your adapter the statement of asm00, greedy, CPU, float32, 200 new tokens. Writes `asm00.s`, builds it, runs it, prints the result. |
| `kit.jsonl` | 40 training examples: 32 programs that print a string, 8 that only exit. None of them is asm00. |
| `prompt.py` | The prompt as the grading server builds it: Qwen3 chat markup, thinking block empty and closed. `QWEN_REVISION` pins the snapshot the grading box caches. |
| `selftest.py` | The basic tests of the graded exercise, locally: adapter, build public, gate. |
| `bootcamp.jsonl` | The bootcamp statements asm00 to asm11, word for word: `id`, `name`, `statement`. Step 1 asks asm00 to asm03. |
| `public.jsonl` | The 11 public statements of the graded exercise, with their cases. They measure; they never train. |
| `requirements.txt` | The versions the reference adapter was made with, plus `datasets` and `huggingface_hub` for the later steps of the course. |
| `KIT-README.md` | This file. |

## Prerequisites

Linux on x86-64, with `nasm` and `ld` (package `binutils`), Python 3.10 or newer and `uv`
(https://docs.astral.sh/uv/getting-started/installation/). Debian and Ubuntu:

```bash
sudo apt install nasm binutils python3
```

## Where to unpack it

At the root of your `asm_llm` repository: `tar xzf asm-llm-kit-minimal.tar.gz --strip-components=1`.

## The commands

```bash
uv run get_model.py                                   # once: Qwen3-0.6B into ./model, never committed
uv run train.py ./model adapter kit.jsonl --device cpu --epochs 1 --lr 5e-4 --rank 8 --seed 0
uv run try.py adapter                                 # asm00, with your adapter
uv run selftest.py ./model --adapter adapter          # the public statements, as the grader replays them
```

`uv run` reads the Python dependencies from the script header and installs them in its own
environment; `nasm` and `ld` come from the prerequisites above. torch comes from the PyTorch CPU index, about 200 MB, not the CUDA build of PyPI.

## Memory and time

Training needs about 5 GB of free memory: our peaks were 4.64 and 5.04 GiB under an 8 GB limit.
With 4 GiB and no swap, it was killed while loading the weights.
Our runs took 82 to 107 s on 4 CPU cores (2026-10-15). `try.py` took 9 s on 2 cores.

## What the grader accepts in `adapter/`

- Exactly two files: `adapter_config.json` and `adapter_model.safetensors`. `train.py`
  deletes the `README.md` that `save_pretrained` writes. Nothing else goes in `adapter/`.
- Plain LoRA for `Qwen/Qwen3-0.6B`: rank 1 to 64, 128 MiB at most, no `use_dora`, no
  `use_rslora`, `bias: "none"`, no `modules_to_save`. The grader applies W + (alpha/r)·B·A and nothing else.

## The prompt: train on the one that is graded

```
<|im_start|>system
{prompt.txt or the default}<|im_end|>
<|im_start|>user
Exercise p01. File: p01.s

{statement}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

This is `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`.
A model trained with the default (`enable_thinking=True`) is not the one that is graded.
