# Caribbean LLM - Capstone Project

## Caribbean AI Academy - Graduates Module
### By Adrian Dunkley

---

## Project Overview

Build a fine-tuned language model dat understands Caribbean culture,
history, languages (English, Creole, Patois), and context.

Di goal: take a small pre-trained language model and adapt it fi
Caribbean-specific knowledge using the techniques from Lessons 01-02.

---

## Project Structure

```
project_caribbean_llm/
├── README.md          # This file
├── main.py            # Main fine-tuning pipeline
├── data_prep.py       # Caribbean text corpus preparation
└── evaluate.py        # Model evaluation scripts
```

---

## Objectives

1. **Data Collection**: Build a Caribbean text corpus from multiple sources
2. **Data Preparation**: Clean, tokenize, and format data for fine-tuning
3. **Fine-Tuning**: Apply LoRA to adapt a small LM (GPT-2 small or similar)
4. **Evaluation**: Test on Caribbean-specific benchmarks
5. **Deployment**: Package for inference

---

## Getting Started

### Prerequisites
```bash
pip install transformers datasets peft accelerate torch
pip install sentencepiece protobuf
```

### Steps

1. Prepare your Caribbean data:
```bash
python data_prep.py
```

2. Run fine-tuning:
```bash
python main.py --epochs 3 --lr 2e-4 --rank 16
```

3. Evaluate:
```bash
python evaluate.py --model_path ./output/caribbean-llm-lora
```

---

## Data Sources (Suggested)

- Caribbean newspaper archives (Jamaica Gleaner, Trinidad Guardian, etc.)
- CARICOM official documents
- UWI academic publications
- Caribbean literature (public domain)
- Caribbean Wikipedia articles
- Hansard (parliamentary records) from Caribbean nations
- Caribbean music lyrics (with appropriate licensing)
- Caribbean recipe collections and cultural guides

---

## Evaluation Criteria

| Metric | Target | Weight |
|--------|--------|--------|
| Caribbean QA Accuracy | > 70% | 30% |
| Cultural Sensitivity Score | > 0.8 | 20% |
| Creole/Patois Understanding | > 60% | 20% |
| Perplexity (lower is better) | < 50 | 15% |
| Response Fluency (human eval) | > 4/5 | 15% |

---

## Caribbean Countries Covered

The model should have knowledge about ALL CARICOM member states:
- Antigua and Barbuda, Bahamas, Barbados, Belize
- Dominica, Grenada, Guyana, Haiti
- Jamaica, Montserrat, St. Kitts and Nevis, St. Lucia
- St. Vincent and the Grenadines, Suriname, Trinidad and Tobago

Plus associate members and other Caribbean nations.

---

## Tips from Adrian

> "Start small - fine-tune on 1000 high-quality QA pairs before
> scaling up. Quality over quantity, same like good rum."

> "Test wid actual Caribbean people. If yuh grandmother nah
> understand di model's answer, it nah good enough."

> "Remember: di goal is not fi replace Caribbean knowledge holders,
> but fi make their knowledge accessible to everyone."
