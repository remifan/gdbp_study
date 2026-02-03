# Generalized DBP - Combined Deep Learning and Adaptive DSP Training

This repository contains the source code and supplementary materials for the paper:

> **Combined Neural Network and Adaptive DSP Training for Long-Haul Optical Communications**
> Qirui Fan, Chao Lu, Alan Pak Tao Lau
> IEEE Journal of Lightwave Technology, 2021
> [DOI: 10.1109/JLT.2021.3111437](https://ieeexplore.ieee.org/abstract/document/9534655)

## Overview

<p align="center">
  <img src="docs/images/fig2.svg" alt="GDBP Architecture" width="700">
</p>
<p align="center"><em>The NN GDBP and adaptive DSP structure: Conv1D (1-D convolution), BPN (batch power normalization), MIMO (multiple-input-multiple-output filters), FOE (frequency offset estimator).</em></p>

We propose a novel "stateful neural network" layer framework that integrates adaptive DSP algorithms with standard batch-based backpropagation training. This approach enables joint optimization of neural network parameters and adaptive filters for optical signal processing.

The architecture is defined in a neural network-like fashion using composable layers:

```python
from commplax.module import core, layer

model = layer.Serial(
    layer.FDBP(steps=steps,
               dtaps=dtaps,
               ntaps=ntaps,
               d_init=d_init,
               n_init=n_init),
    layer.BatchPowerNorm(mode=mode),
    layer.MIMOFOEAf(name='FOEAf',
                    w0=w0,
                    train=mimo_train,
                    preslicer=core.conv1d_slicer(rtaps),
                    foekwargs={}),
    layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),  # R-filter
    layer.MIMOAF(train=mimo_train))  # adaptive MIMO equalizer
```

For a detailed walkthrough with reproducible results, see our [extended web article](https://remifan.github.io/gdbp_study/article.html).

## Dataset

This study uses the [LabPtPTm2](https://github.com/remifan/LabPtPTm2) dataset: experimental data from 1125 km 7-channel DP-16QAM WDM transmission using quantum random number generated bit sequences. The dataset is hosted on AWS S3 with Python APIs for programmatic access.

[![DOI](https://img.shields.io/badge/DOI-10.6084/m9.figshare.14843037-blue)](https://doi.org/10.6084/m9.figshare.14843037)

## Dependencies

This project depends on [commplax](https://github.com/remifan/commplax), a JAX-based differentiable DSP library for optical communications.

**Important:** This code is compatible with [commplax v0.1.1](https://github.com/remifan/commplax/tree/v0.1.1). Later versions of commplax have undergone significant changes and may not be compatible.

The required package versions (pinned in commplax v0.1.1):
| Package | Version |
|---------|---------|
| Python  | 3.8     |
| jax     | 0.2.13  |
| jaxlib  | 0.1.66  |
| flax    | 0.3.4   |

## Installation

Since the dependencies are outdated, we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create an isolated environment.

### 1. Create conda environment with Python 3.8

```bash
conda create -n gdbp python=3.8 -y
conda activate gdbp
```

### 2. Install JAX (CPU version)

```bash
pip install jax==0.2.13 jaxlib==0.1.66
```

> **GPU support:** The CUDA-enabled jaxlib 0.1.66 wheels are likely stale and may not work with modern CUDA drivers. If you need GPU acceleration, refer to the [JAX installation guide](https://github.com/google/jax#installation) for that era. The general pattern was:
> ```bash
> pip install jaxlib==0.1.66+cudaXXX -f https://storage.googleapis.com/jax-releases/jax_releases.html
> ```
> where `XXX` corresponds to your CUDA version (e.g., `cuda110` for CUDA 11.0).

### 3. Install commplax v0.1.1 and gdbp_study

```bash
pip install git+https://github.com/remifan/commplax.git@v0.1.1
git clone https://github.com/remifan/gdbp_study.git
cd gdbp_study
pip install -e .
```

## Citation

If you find this work useful, please cite:
```bibtex
@article{fan2021combined,
  title={Combined Neural Network and Adaptive DSP Training for Long-Haul Optical Communications},
  author={Fan, Qirui and Lu, Chao and Lau, Alan Pak Tao},
  journal={Journal of Lightwave Technology},
  volume={39},
  number={22},
  pages={7083--7091},
  year={2021},
  publisher={IEEE},
  doi={10.1109/JLT.2021.3111437}
}
```