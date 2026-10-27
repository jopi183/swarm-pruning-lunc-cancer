# Constrained MOPSO for Sensitivity-Aware Pruning of Lung Cancer Histopathology Networks

Official implementation of the paper *Constrained Multi-Objective Particle Swarm Optimization for Sensitivity-Aware Pruning of Lung Cancer Histopathology Networks* (MethodsX)

## Overview
Four-stage pipeline for layer-wise pruning of ResNet-50 with per-class sensitivity preservation:
1. Dense ResNet-50 baseline training
2. Per-layer zero-shot sensitivity analysis producing data-driven sparsity bounds
3. Constrained MOPSO search with zero-shot proxy fitness
4. Fine-tuning of three selected operating points from the Pareto archive

## Two implementation variants
This repository provides two parallel implementations of the MOPSO search that differ in their archive management strategy. Both share identical dataset handling, backbone, sensitivity analysis, proxy fitness, operating point selection, and fine-tuning code.


## Repository structure

```
.
├── mopso-pipeline.ipynb          
├── baselines-and-ablations.ipynb    
├── README.md
├── LICENSE
├── data/
│   ├── LungHist700/
│   └── LC25000/
├── checkpoints/
└── results/
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer with CUDA 11.8 or newer
- NVIDIA GPU with at least 8 GB VRAM (tested on RTX 4060)



## Datasets

Two public datasets are used; neither is redistributed in this repository.

### LungHist700

Diosdado et al. (2024), *Scientific Data* 11:1088. Download from [https://doi.org/10.6084/m9.figshare.26104051](https://doi.org/10.6084/m9.figshare.26104051).

Expected structure under `data/LungHist700/`:

```
data/LungHist700/
├── aca_bd/   # adenocarcinoma, well differentiated
├── aca_md/   # adenocarcinoma, moderately differentiated
├── aca_pd/   # adenocarcinoma, poorly differentiated
├── scc_bd/   # squamous cell carcinoma, well differentiated
├── scc_md/   # squamous cell carcinoma, moderately differentiated
├── scc_pd/   # squamous cell carcinoma, poorly differentiated
└── nor/      # normal lung tissue
```

The seven subclasses are grouped into three superclasses (aca, scc, nor) by the notebooks.

### LC25000 (lung subset)

Borkowski et al. (2019), [arXiv:1912.12142](https://arxiv.org/abs/1912.12142). Download from [https://github.com/tampapath/lung_colon_image_set](https://github.com/tampapath/lung_colon_image_set) and extract only the lung images (three classes).

```
data/LC25000/
├── lung_aca/
├── lung_n/
└── lung_scc/
```

## Usage

### Single-seed run (either variant)

For the canonical variant:

```python
SEED = 42
DATASET = 'LungHist700'
```
### Multi-seed statistics

Rerun with `SEED` in `{42, 123, 456, 789, 2026}`. Each seed writes to a separate folder. 


## Reproducibility notes

- All stochastic components are controlled by a single `SEED` constant.
- The data split uses `torch.Generator().manual_seed(SEED)` passed explicitly to `random_split`, so split indices are independent of the order of other RNG calls.
- `DETERMINISTIC_CUDNN = True` by default. Set to `False` during development for faster but non-reproducible runs.
- Both variants save intermediate results after every experiment, so a crashed session can be resumed by rerunning only the remaining cells.

## Implementation notes

### Baselines

Notebook 02 (either variant) implements six baseline methods under the shared training pipeline:

- Global magnitude at `s ∈ {0.3, 0.5, 0.7}`
- Global Magnitude Pruning (GMP) at `s ∈ {0.5, 0.7}`
- SNIP at `s ∈ {0.3, 0.5, 0.7}`

Baselines do not use MOPSO internals and therefore produce the same results under either variant for the same seed and dataset.

### Ablations
Three ablation studies compare against the proposed method:
- Ablation 1: MOPSO with vs. without medical sensitivity constraint
- Ablation 2: Multi-objective MOPSO vs. single-objective PSO
- Ablation 3: Proxy fitness vs. full-evaluation MOPSO (compute cost)

The ablation MOPSO is implemented as a flagged version of the main MOPSO class in each variant.


