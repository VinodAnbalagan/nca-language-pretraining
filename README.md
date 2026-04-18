# NCA Language Pre-Training — Interactive Exploration

> _What if GPT learned to think before it learned to speak?_

An interactive marimo notebook and empirical investigation extending
**"Training Language Models via Neural Cellular Automata"**
(Lee, Han, Kumar & Agrawal, MIT — [arXiv:2603.10055](https://arxiv.org/abs/2603.10055)).

Built for the [marimo × alphaXiv Notebook Competition](https://marimo.io/pages/events/notebook-competition) (April 2026).

---

## What This Is

The paper shows that pre-training a transformer on **Neural Cellular Automata (NCA) dynamics** — before any natural language — improves downstream language modelling and reasoning. 164M NCA tokens outperform 1.6B tokens of Common Crawl.

This repo reproduces that core finding at toy scale and extends it with two original contributions:

| Contribution                           | What we ask                      | What we test                                         |
| -------------------------------------- | -------------------------------- | ---------------------------------------------------- |
| **Extension A: 1D NCA**                | Does the geometry need to be 2D? | 1D tape NCA vs 2D grid NCA on the same transfer task |
| **Extension B: Complexity Curriculum** | Does order of complexity matter? | Random rule sampling vs simple→complex curriculum    |

Both extensions are CPU-feasible, empirically honest, and directly challenge assumptions the paper never ablated.

---

## Results (Dyck-1 Transfer Task, 150k-param Transformer)

> Run `python experiments/day_1_test.py` to reproduce.

| Condition                     | Final Val Loss | Steps to Convergence | vs Scratch |
| ----------------------------- | -------------- | -------------------- | ---------- |
| A: Scratch                    | —              | —                    | baseline   |
| B: 2D NCA (paper)             | —              | —                    | —          |
| C: 1D NCA (ours)              | —              | —                    | —          |
| D: 2D NCA + Curriculum (ours) | —              | —                    | —          |

_Results populate after running the experiment. Loss curves saved to `results/loss_curves.png`._

---

## Repo Structure

```
nca-language-pretraining/
│
├── experiments/
│   └── day_1_test.py        # Raw experiment script — run this first
│                            # 4 conditions × 3 seeds, ~5 min on CPU
│                            # Answers: does NCA pre-training help at toy scale?
│
├── notebook/
│   └── nca_notebook.py      # Marimo competition notebook (coming)
│                            # Interactive, type-along, deployable to molab
│
├── assets/                  # Figures, animations (generated at runtime)
├── results/                 # Experiment outputs (loss curves, tables)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/nca-language-pretraining.git
cd nca-language-pretraining

# 2. Environment
python -m venv nca_env
source nca_env/bin/activate      # Mac/Linux
# nca_env\Scripts\activate       # Windows

# 3. Install
pip install -r requirements.txt

# 4. Run the experiment
python experiments/day_1_test.py

# Expected output:
# ============================================================
# RESULTS SUMMARY
# Condition                      Final Val Loss   Steps to <0.3
# ------------------------------------------------------------
# A: Scratch                     X.XXXX ± X.XXXX          >20
# B: 2D NCA (paper)              X.XXXX ± X.XXXX            X
# C: 1D NCA (ours)               X.XXXX ± X.XXXX            X
# D: 2D NCA + Curriculum (ours)  X.XXXX ± X.XXXX            X
# Loss curves saved to: results/loss_curves.png
# Total runtime: X.X minutes
```

---

## The Paper's Core Idea

```
Stage 0  ──▶  NCA Pre-Pre-Training
              12×12 grid of cells, each with n discrete states
              Random neural net = update rule (different per sequence)
              Transformer learns: given grid at time t, predict grid at t+1
              164M tokens, CPU-feasible generation

Stage 1  ──▶  Language Pre-Training
              Transfer attention weights from Stage 0
              Re-initialise embeddings for natural language vocabulary
              Same compute budget as a scratch baseline

Stage 2  ──▶  Downstream Evaluation
              GSM8K (math), HumanEval (code), BigBench-Lite (reasoning)
              NCA-init outperforms scratch by up to 6%
```

**Key finding:** The transferable priors live in the **attention layers**, not the MLPs.
NCA training forces the transformer to do **in-context rule inference** — observe a sequence,
hypothesise the hidden rule, apply it forward. This is exactly what LLMs do with language.

---

## Our Extensions

### Extension A — 1D NCA

The paper uses 2D grids because NCAs are classically 2D. But language is **1D**.
We ask: does the geometry need to be 2D, or is a 1D tape sufficient?

```
1D NCA:  tape of L cells, each with n states
         neighbourhood: left, self, right (3 inputs instead of 9)
         tokenisation: window of 2 cells → 1 token (vocab = n² instead of n⁴)
         sequences: structurally closer to language tokens
```

If 1D NCA matches or beats 2D NCA on transfer, that's a significant finding —
the spatial structure the paper takes for granted may not be necessary.

### Extension B — Complexity Curriculum

The paper samples NCA rules randomly, filtered by gzip complexity band.
We ask: does the **order** of complexity exposure matter?

```
Random (paper):     rules sampled uniformly from the complexity band
Curriculum (ours):  epoch 1 → simplest 30% of rules
                    epoch 2 → middle 40% of rules
                    epoch 3 → all rules (consolidation)
```

Curriculum learning has strong precedent in the broader ML literature.
Applied here, it means the transformer builds simple dependency-tracking circuits
before being exposed to the chaotic dynamics that stress-test them.

---

## The Transfer Task: Dyck-1

We use **balanced parentheses** as the transfer task.

```
Valid:    ((()))    (()())    ()
Invalid:  )(        (()       ))((
```

Why Dyck-1?

- Requires genuine long-range dependency tracking (every `(` must find its `)`)
- Generates instantly — no downloads, no data pipeline
- Clean convergence signal — loss drops clearly when the model "gets it"
- Classic benchmark for sequential reasoning in formal language theory

---

## Connection to ARC-AGI

Every NCA sequence is a miniature ARC puzzle:

> _Observe the pattern. Infer the hidden rule. Apply it forward._

This is precisely what ARC-AGI tasks require. NCA pre-pre-training may be building
the same in-context rule inference circuits that abstract reasoning demands.
We explore this connection in the marimo notebook (Section 5).

---

## Citation

If you build on this work:

```bibtex
@misc{lee2026traininglanguagemodelsneural,
      title={Training Language Models via Neural Cellular Automata},
      author={Dan Lee and Seungwook Han and Akarsh Kumar and Pulkit Agrawal},
      year={2026},
      eprint={2603.10055},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.10055},
}
```

---

## About

Built by [Vinod Anbalagan](https://substack.com/@vinodanbalagan) for the
marimo × alphaXiv Notebook Competition (April 2026).

Part of ongoing research into geometric and object-centric approaches to
abstract reasoning — see [The Meta Gradient](https://substack.com/@vinodanbalagan) on Substack.

_"We didn't teach this model to think by showing it thoughts.
We taught it to think by showing it patterns.
Language was never the point — structure was."_
