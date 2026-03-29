# FlashAttention Speedup Under Multi-GPU Communication Overhead: DDP vs FSDP

CISC-727: Advanced Research Explorations (ARE) - I | Assignment A04

**Author:** Kenneth Peter Fernandes
**Institution:** Harrisburg University of Science and Technology
**Instructor:** Professor Majid Shaalan, PhD

## Research Question

Does FlashAttention's speedup over standard attention diminish as communication overhead increases when scaling from single-GPU to multi-GPU training using DDP and FSDP?

## A04 Lever: Partition Granularity (patch_size 16 -> 4)

A03 showed FlashAttention had zero speedup because ViT-Small with patch_size=16 on CIFAR-10 produces only 5 tokens (the 5x5 attention matrix fits entirely in SRAM). This assignment changes one lever — **partition granularity** — by reducing patch_size from 16 to 4, increasing sequence length from 5 to 65 tokens.

## Key Finding

FlashAttention is **16.34% slower** than standard attention at 65 tokens on T4 GPUs under DDP — the opposite of the predicted speedup. At 5 tokens (A03 baseline), the difference was noise (+0.48%). This indicates 65 tokens is still below FlashAttention's efficiency crossover threshold on T4 hardware.

| Run ID | patch_size | Seq Len | Attention | Median Time/Step (s) | Throughput (samples/s) |
|--------|-----------|---------|-----------|---------------------|----------------------|
| R1-baseline-std | 16 | 5 | math | 0.0601 | 2131.3 |
| R1-baseline-flash | 16 | 5 | flash | 0.0598 | 2141.5 |
| R2-variant-std | 4 | 65 | math | 0.2209 | 579.5 |
| R2-variant-flash | 4 | 65 | flash | 0.2570 | 498.1 |

**Decision: Narrow** — switch to a long-sequence NLP workload (512+ tokens) in ARE-II to find the crossover point.

## Repository Structure

```
.
├── configs/
│   ├── baseline.yaml              # A03 baseline: patch_size=16 (Run ID: R1-baseline)
│   └── variant.yaml               # A04 variant: patch_size=4 (Run ID: R2-variant)
├── data/
│   └── dist_train_a04.py          # Distributed training script (launched via torchrun)
├── docs/
│   ├── a02/                       # Literature review (A02)
│   │   ├── *.tex / *.pdf
│   │   └── references.bib
│   ├── a03/                       # Pilot baseline (A03)
│   │   ├── evaluation_protocol/   # Evaluation protocol
│   │   ├── notebook/              # A03 experiment notebook
│   │   └── proposal_seed/         # Proposal seed document
│   └── a04/                       # Strong result deliverables (A04)
│       ├── interpretation/        # Deliverable 2: Interpretation (1-2 pages)
│       │   ├── interpretation.tex
│       │   └── interpretation.pdf
│       ├── decision_memo/         # Deliverable 3: Decision memo (1 page)
│       │   ├── decision_memo.tex
│       │   └── decision_memo.pdf
│       └── ai_usage_log.csv       # AI tools usage documentation
├── notebook/
│   └── a04_strong_result.ipynb    # Deliverable 1: Experiment notebook (Kaggle)
├── results/                       # Saved experiment outputs
│   ├── all_results.json           # Full results with per-rep data
│   ├── summary.csv                # Summary table
│   ├── primary_figure.png         # Primary figure (Deliverable 1)
│   └── supporting_breakdown.png   # Supporting figure: fwd/bwd breakdown
├── .gitignore
└── README.md
```

## How to Reproduce the Strong Result

### Environment

- **Platform:** Kaggle Notebooks (GPU accelerator: T4 x2)
- **GPUs:** 2x NVIDIA Tesla T4 (16 GB HBM2 each)
- **Interconnect:** PCIe
- **Framework:** PyTorch 2.10.0+cu128
- **Dataset:** CIFAR-10 — added as Kaggle input from [cifar-10-python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python) (170.5 MB)

### Step 1: Set Up Kaggle Notebook

1. Create a new Kaggle notebook
2. Under **Settings**, set Accelerator to **GPU T4 x2**
3. Click **Add Input** in the sidebar and search for `cifar-10-python` ([link](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python)) — this mounts the dataset at `/kaggle/input/cifar-10-python/`
4. Upload or paste `notebook/a04_strong_result.ipynb`

### Step 2: Run All Cells

The notebook runs 4 configurations x 3 repetitions = 12 runs automatically:

| Run ID | patch_size | Seq Len | Attention | Description |
|--------|-----------|---------|-----------|-------------|
| R1-baseline-std | 16 | 5 | math | A03 baseline, standard attention |
| R1-baseline-flash | 16 | 5 | flash | A03 baseline, FlashAttention |
| R2-variant-std | 4 | 65 | math | A04 variant, standard attention |
| R2-variant-flash | 4 | 65 | flash | A04 variant, FlashAttention |

Each run: batch_size=64/GPU, 100 steps, 10 warmup, seed=42, float32, AdamW lr=1e-3

### Step 3: Collect Results

After execution, the notebook saves:
- `results/summary.csv` — summary table
- `results/all_results.json` — full results with per-rep timing data
- `results/primary_figure.png` — main comparison figure
- `results/supporting_breakdown.png` — forward/backward/comm breakdown

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| D1 | Primary Figure/Table | `results/primary_figure.png` and `notebook/a04_strong_result.ipynb` |
| D2 | Interpretation (1-2 pages) | `docs/a04/interpretation/interpretation.pdf` |
| D3 | Decision Memo (1 page) | `docs/a04/decision_memo/decision_memo.pdf` |
| D4 | Walkthrough Recording | Submitted separately via LMS |

