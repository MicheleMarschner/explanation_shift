# Explanation Shift under Domain Shift

This project studies how model explanations change when the input distribution shifts. Using **CIFAR-10** as the clean reference distribution and **CIFAR-10-C** as a controlled benchmark for corrupted inputs, it investigates whether explanation methods remain stable as distribution shift becomes stronger.

More specifically, the project analyzes the relationship between **distribution shift**, **predictive behaviour**, and **explanation drift**. It compares clean and corrupted inputs across multiple corruption types, severity levels, random seeds, and explainers in order to examine when explanations change together with model performance and when they do not.

> Disclaimer
> AI tools were used exclusively for bug fixing and improving readability, including class and function documentation, visualization styling, and docstrings. They were not used for full code generation or for modeling decisions. 

---

## Repository Structure

The repository has the following high-level structure:
```text
EXPLANATION_UNDER_SHIFT/
├── .vscode/                     
├── checkpoints/                 # Saved resnet model checkpoints, trained on CIFAR-10
├── configs/                     # Configs for running experiments
├── data/
│   ├── CIFAR-10/               # 
│   └── CIFAR-10-C/             # 
├── experiments/                # Experiment outputs, run folders, and intermediate artifacts
├── report.pdf                  # Final report
├── results/                    # Processed results, plots, tables, and evaluation outputs
├── slurm_scripts/              # HPC / SLURM job example scripts for cluster execution
├── src/                        # Main project source code
├── .gitattributes              
├── .gitignore                  
├── pyproject.toml              # Project/package configuration and dependencies
├── README.md                   # Project overview and usage instructions
└── requirements-captum-nodeps.txt # Optional dependency list for Captum-related setup
```

---

## Setup Guide

The project was developed in a virtual environment called `shift`. Using the same environment name and package versions is the easiest way to reproduce the original results. The instructions below use **conda**, but a very similar setup should also work with other environment managers as long as the same Python version and dependency versions are used.

### 1. Install Conda

If you do not already have conda installed, the easiest option is usually **Miniconda** or **Anaconda**.

#### macOS / Linux

Download and install **Miniconda** for your operating system and architecture from the official website.

After installation, initialize conda for your shell if needed:

```bash
conda init zsh
source ~/.zshrc
```

For other shells, replace `zsh` with the shell you use, for example `bash`.

Restart the terminal after installation if needed.

#### Windows (PowerShell)

Download and install **Miniconda** for Windows from the official website. During installation, allow it to set up shell integration if prompted.

After installation, open a new PowerShell terminal and verify that conda is available:

```powershell
conda --version
```

### 2. Create and Activate the Environment

Create the environment with Python 3.11:

```bash
conda create -n shift python=3.11
conda activate shift
```

Install the project in editable mode:

```bash
pip install -e .
```

Install the additional Captum-related requirements:

```bash
pip install -r requirements-captum-nodeps.txt
```

This setup installs the project dependencies defined in `pyproject.toml` together with the additional requirements used in the interpretability pipeline.

### 3. Verify the Installation

Check that the package and key dependencies are available:

```bash
python -m pip show shift-project
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); import fiftyone as fo; print('FiftyOne installed')"
```

If you use VS Code or Jupyter, make sure the `shift` kernel is selected. If it does not appear automatically, register it manually:

```bash
python -m ipykernel install --user --name shift --display-name "Python 3.11 (shift)"
```

---

## Data

This project uses **CIFAR-10** as the clean in-distribution dataset and **CIFAR-10-C** as the corrupted benchmark for evaluating model and explanation behaviour under distribution shift.

The data is expected under the `data/` directory:
- `data/CIFAR-10/`
- `data/CIFAR-10-C/`

If the datasets are not already present locally, download them from the official sources:
- **CIFAR-10:** [official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)
- **CIFAR-10-C:** [Zenodo record](https://zenodo.org/records/2535967)

After downloading, extract the files so that the folder structure matches the paths above.

---

## Model Architecture and Weights

The experiments use the **ResNet** architecture implemented in `src/resnet.py`.
The implementation and pretrained weights are based on:
- [PyTorch_CIFAR10 by huyvnphan](https://github.com/huyvnphan/PyTorch_CIFAR10)

Place the pretrained model weights in the `checkpoints/` directory so they can be loaded by the pipeline.

---

## Running Experiments

The experiment grid is defined in `src/configs/experiments_config.py` and can be adapted for consecutive experiments. 
This file specifies the global settings used to generate all runs in the pipeline:

- **`N_PAIRS = 1000`**: number of clean–corrupted image pairs evaluated per run
- **`CORRUPTIONS = ["gaussian_noise", "defocus_blur", "brightness", "fog"]`**: corruption types included
- **`SEVERITIES = [1, 2, 3, 5]`**: corruption severity levels evaluated
- **`EXPLAINERS = ["IG", "GradCAM"]`**: explanation methods used 
- **`SEEDS = [7, 42, 52, 128, 1200]`**: random seeds used to test robustness

Example command:

```bash
PYTHONPATH=src python -m src.main \
  --config src/configs/experiments_config.py \
  --stage drift
```

The commands need to be started from the project root directory. 

---

## Running Analyses

After training runs are complete, the corresponding analysis scripts can be executed to recreate summary tables, comparisons, and visualizations.

```bash
PYTHONPATH=src python -m src.main --mode analysis
```

**Note**: 
The container will automatically attempt to download the experiments archive if no local experiment folders are present.

If the automatic download fails, manually download the archive from [here](https://drive.google.com/file/d/1YGeHsA141o4Rfwor1sXd0_sdwooDRIT4/view?usp=sharing), ensure it is available as a .tar.gz file, and extract it in the project root.

---

## Outputs and artifacts

### Per experiment run (stored in `./experiments/<run_name>/`)
Each experiment creates one folder containing all artifacts needed to reproduce the run and analyse it later:

!TODO auf hoher flugebene eher was in den einzelnen Ordnern gesammelt drinnen ist - oder soll .pt structure erklärt werden. 

### Results (stored in `./results/`)

!TODO auf hoher flugebene eher was in den einzelnen Ordnern gesammelt drinnen ist

---

## Limitations

