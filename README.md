# Bardic Fine-Tuning

## Quickstart
- Install toolchain (macOS): `uv venv && source .venv/bin/activate && uv pip install --upgrade pip`
- Install deps: `uv pip install -e .`
- Launch Jupyter: `uv run python -m ipykernel install --user --name bardic-env --display-name "bardic-env"`
- Start notebooks: `uv run jupyter lab`

## Notebooks
- 01_Data_Generator.ipynb — Generate/refresh `data/poetic_refusal.jsonl` via OpenAI/Anthropic or stubbed bardic refusals.
- 02_Trainer_Arena.ipynb — Sequential LoRA then DoRA fine-tunes with Unsloth, saving adapters under `outputs/`.
- 03_Judge_Arena.ipynb — Load both adapters, run side-by-side refusals, and compare outputs.

## Data
- Seed file: `data/poetic_refusal.jsonl` with system/user/assistant rows in chat format.

## Notes
- Core stack: unsloth, transformers, trl, peft, datasets, torch, accelerate, wandb (optional), ipywidgets.
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your env to hit live providers in the data generator.
