# asd-ai-scoping-review

This repository contains data processing code for the scoping review paper
"Systematic Scoping Review of AI Applications for Automatic Autism Assessment using Behavioral Data".

## Repository structure

```
data/       # input data files
scripts/    # processing and analysis scripts
output/     # generated results and figures
```

## Setup

Use [conda](https://docs.conda.io/) or [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to create the environment:

```bash
# conda
conda env create -f environment.yaml
conda activate asd-scoping-review

# micromamba
micromamba env create -f environment.yaml
micromamba activate asd-scoping-review
```
