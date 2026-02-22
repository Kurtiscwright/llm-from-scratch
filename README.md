# LLM From Scratch

Working implementation of a transformer language model, built from scratch following
*Build a Large Language Model From Scratch* by Sebastian Raschka.

## Structure

- `src/` — Core implementation (tokenizer, attention, transformer blocks, training loop)
- `notebooks/` — Exploratory work and chapter exercises
- `data/` — Training data (gitignored if large)
- `tests/` — Unit tests for core components

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Hardware

Developed on Apple Silicon (MPS backend). Cross-architecture experiments on NVIDIA (CUDA)
and AMD (ROCm) GPUs documented separately.
