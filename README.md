# Bardic Fine-Tuning: LLM as Poetic Conversationalist

## Executive Summary

Successfully fine-tuned Mistral Nemo 12B to respond exclusively in poetic format while maintaining conversational capabilities. **Best configuration (DoRA, 32/64, 1k samples) achieved 2.80/5.0 quality score, 82% format adherence, and only 2% failure rate** compared to 0.70 quality and 56% failure rate on the base model.

### Key Achievements
| Metric | Base Model | Best LoRA | Best DoRA |
|--------|-----------|-----------|----------|
| **Quality Score** | 0.70 | 2.62 | **2.80** ✨ |
| **Format Adherence** | 8.0% | 80.0% | **82.0%** |
| **Failure Rate** | 56.0% | 6.0% | **2.0%** |

## Critical Findings

1. **DoRA > LoRA for Language Tasks** — DoRA configurations consistently outperformed LoRA equivalents, validating its superior learning patterns for complex linguistic tasks (Liu et al., 2024).

2. **Overfitting at Scale** — Training on full 7k dataset caused catastrophic repetition despite lower validation loss. Early stopping at 1k samples yielded better qualitative results.

3. **System Prompt Necessity** — Base model's pre-existing conversational priors are strong; fine-tuning teaches the model to *associate* poetic behavior with system prompt context rather than *replacing* default behavior.

4. **Factual Knowledge Degradation** — Current dataset is ~90% poetic, ~10% factual. Factual responses diminished after fine-tuning; increasing factual data proportion is recommended.

5. **Hyperparameter Sensitivity** — Combining all regularizations (5% dropout, 0.01 decay, 5e-4 LR) on full dataset underperformed, indicating DoRA is sensitive to overfitting.

## Configuration & Training

### Best Performing Model
- **Adapter**: DoRA (Weight-Decomposed LoRA)
- **Rank/Alpha**: 32/64
- **Learning Rate**: 2e-4
- **Scheduler**: Cosine
- **Dropout**: None
- **Weight Decay**: 0.001
- **Dataset**: 1k samples (early-stopped)
- **Epochs**: 1

### Dataset Composition
- **Base**: Poem Comprehensive Dataset (PCD) + Mistral Small Creative generated prompts
- **Expansion**: LIMA (wikihow subset) + NoRobots (Open QA, Closed QA, Summarize)
- **Total Training**: ~7k samples refined, with ~1k used for best model
- **Validation**: Held-out 50 prompts for evaluation

## Notebooks

- **01_Data_Generator.ipynb** — Generate synthetic poem dataset via OpenAI/Anthropic
- **01_Data_Refiner.ipynb** — Expand and refine dataset with varied-length poems and real conversations
- **02_Trainer.ipynb** — Sequential fine-tuning loop (LoRA & DoRA configs) with validation monitoring
- **03_Judge_Arena.ipynb** — Side-by-side inference & qualitative comparison
- **04_Evaluation.ipynb** — Quantitative evaluation and quality distribution analysis

## Environment & Hardware

- **Base Model**: unsloth/Mistral-Nemo-Base-2407 (quantized, 12B)
- **Framework**: Unsloth + Transformers + PEFT + TRL
- **Hardware**: Windows 11 RTX 4070 Ti or GPU cluster (20GB+ VRAM for larger configs)
- **Environment Manager**: `uv`

## Artifacts & Results

### Dataset Location
- **Raw Data**: `data/` folder
  - `lima_train.jsonl` — LIMA dataset (filtered to wikihow subset)
  - `poem_condense.csv` — Initial poetry source
  - `poem_finetune_13000.jsonl` — Generated poem dataset with queries
  - `poem_refined_2800x6.jsonl` — Expanded, multi-variant poem dataset (training data)
  - `poem_real_conversations_2000.jsonl` — Converted real conversations from LIMA & NoRobots to poetic format

### Sample Inferences & Outputs
- **Trained Adapters**: `outputs/` folder (does not include everything as the sizes are too large)
- **Evaluation Outputs**: `results/` folder
  - `eval_summary.json` — Quantitative metrics for all 12 configurations
  - `eval_judge_arena_*.csv` — Detailed per-prompt ratings for each model
  - `quality_dist_judge_arena_*.png` — Quality distribution charts (12 configurations)
  - `training_metrics_dashboard.png` — Training loss and validation trends

### LLM-as-Judge Results Summary
- **Evaluation Methodology**: Automated LLM-based quality scoring (0-5 scale) on 50 held-out prompts
- **Key Results**:
  - Base model: **0.70 quality, 56% failure rate** (poor baseline)
  - Best LoRA (32/64, 1k): **2.62 quality, 6% failure rate**
  - Best DoRA (32/64 Optimised, 1k): **2.80 quality, 2% failure rate** ✨ **[BEST PERFORMER]**
- **Scoring Rubric**: 0 (Gibberish/Repetition) → 5 (Excellent creative poetry with analysis)
- **Full Report**: See `report.pdf` for detailed analysis, methodology, and visual appendix with distribution charts

## Important Notes

- Set `OPEN_ROUTER_API_KEY` in environment for dataset generation
- All training included validation loss monitoring; best checkpoint selected automatically
- Results dashboard available in `results/training_metrics_dashboard.png`
- Quality distribution visualizations for all 12 configurations in appendix of full report

## Ethical Considerations

Fine-tuning for exclusive poetic output introduces safety risks: harmful information can be masked by poetic language, bypassing standard safety filters. Recommend future work on explicit refusal capabilities and safety-aligned fine-tuning.
