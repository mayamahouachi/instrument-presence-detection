<div align="center">

# Music Source Instant Detection Project

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<a href="https://mlflow.org/docs/latest/ml/deep-learning/pytorch/"><img alt="MLFlow" src="https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue"></a>

</div>

## Description

An experimental project about AI model for music source instant detection.

## Installation

### Pip

```bash
# clone project
git clone https://github.com/Convolutio/music-source-sep-structured
cd music-source-sep-structured

# [OPTIONAL] create conda/mamba environment
conda create -n music-sep python=3.13
conda activate music-sep

# adapt the pytorch installation to your gpu capabilities
# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Conda/Mamba

```bash
# clone project
git clone https://github.com/Convolutio/music-source-sep-structured
cd music-source-sep-structured

# create conda/mamba environment and install dependencies
# (mamba is faster)
mamba env create -f environment.yaml -n music-sep

# activate conda environment
mamba activate music-sep
```

### Development workspace

Enable the pre-commit analyses

```sh
# run this once in your repository
# inside your virtual environment
pip install pre-commit
pre-commit install
```

### Install the dataset

1. Be sure to have two directories `train` and `test` with `.stem.mp4` files
   (the raw MUSDB18) in `./data/MUSDB18/raw/`

2. In your environment, run the `build-dataset.py` script:

   ```sh
   python build_dataset.py \
     --musdb-root ./data/MUSDB18/raw/ \
     --out-dir ./data/MUSDB18/prepared/ \
      --sr 22050 --win-sec 1.0 --hop-sec 0.5 \
      --n-mels 64 --n-fft 1024 --mel-hop 256 \
      --ratio-thr-vocals 0.07 \
      --ratio-thr-drums 0.12 \
      --ratio-thr-bass 0.12 \
      --min-mix-rms 1e-4 \
      --stem-abs-thr 1e-3
   ```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
