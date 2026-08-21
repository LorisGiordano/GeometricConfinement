# SurfConDatasets

This repository collects several medical imaging datasets and helper scripts used to inspect, reorganize, and analyze them for geometric confinement and related experiments.

Please follow the instructions to:
1. Create structured datasets
2. Analyse the content of several datasets
3. Create the final MSD datasets that can be used for training.

## Structure

### 1. Dataset structuring

All datasets have a script that can run after downloading the raw dataset, a snapshot of the dataset and a confinement configuration. 
Download the dataset using the provided links at the bottom of this page, and place it in the according directory under the name `original`. 
Then run the following command for the dataset you want to preprocess.

- [Dataset001_CTPel](./Dataset001_CTPel/ctpel.py)
```bash
python Dataset001_CTPel/ctpel.py --input Dataset001_CTPel/original
```
- [Dataset002_HumanInnerEarAnatomy](./Dataset002_HumanInnerEarAnatomy/hiea.py)
```bash
python Dataset002_HumanInnerEarAnatomy/hiea.py --input Dataset002_HumanInnerEarAnatomy/original
```
- [Dataset003_REFUGE2](./Dataset003_REFUGE2/refuge2.py)
```bash
python Dataset003_REFUGE2/refuge2.py --input Dataset003_REFUGE2/original
```
- [Dataset004_VerSe20](./Dataset004_VerSe20/verse20.py)
```bash
python Dataset004_VerSe20/verse20.py --input Dataset004_VerSe20/original
```

### 2. [datasets_analysis.py](./datasets_analysis.py)

Use this script to inspect one dataset or all dataset folders from the repository root.
- reports shape, spacing, and intensity summary statistics
- can target raw dataset folders or `structured/` subfolders when available

```bash
python datasets_analysis.py \
      --dataset Dataset001_CTPel \
      --dataset Dataset002_HumanInnerEarAnatomy
```

### 3. [split_structured.py](./split_structured.py)

This script expects a dataset in the following normalized structure:

```text
structured/
  images/
  labels/
    segmentation/
    landmark/
```

It creates `ImagesTr`, `ImagesVa`, `ImagesTs` and matching `Labels*` folders using a default `70/15/15` split.

```bash
python split_structured.py --input Dataset001_CTPel/structured
```

## Environment

Python version 3.12

### Conda

```bash
conda create -n confinement_datasets python=3.12
conda activate confinement_datasets
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Python venv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset Sources

The external sources listed in `Datasets.txt` are:

- REFUGE2: [Kaggle](https://www.kaggle.com/datasets/victorlemosml/refuge2?resource=download)
- VerSe: [GitHub](https://github.com/anjany/verse)
- Human Inner Ear Anatomy: [Zenodo](https://zenodo.org/records/8277159)
- CTPel: [AIDA Data Hub](https://datahub.aida.scilifelab.se/10.23698/aida/ctpel)
