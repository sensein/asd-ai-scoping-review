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

The Python requirements include spreadsheet support through `openpyxl`. The
PRISMA PDF extractor additionally requires Node dependencies from `package.json`:

```bash
pip install -r requirements.txt
npm install
```

Review workbooks and generated outputs are intentionally excluded from version
control. Before reproducing the analyses, place `final_annotation_sheet*.xlsx`
and `ICR.xlsx` under `data/`, or point `ASD_REVIEW_DATA_ROOT` to a private data
directory. Generated results are written to `output/` (or to
`ASD_REVIEW_OUTPUT_ROOT`). Importing the shared modules and running the unit
tests do not require the private workbooks.

## Research questions

The canonical manuscript labels are:

1. **Participants** — What are the characteristics of participants included in AI-based autism prediction studies using behavioral data?
2. **Study design** — How are AI-based autism prediction studies using behavioral data designed, conducted, and reported?
3. **Behaviors** — How is behavioral data conceptualized and used in AI-based autism prediction?
4. **AI techniques** — How are AI and machine learning techniques applied to behavioral data for autism prediction?
5. **Paper writing and publishing trends** — How has the literature on AI-based autism prediction using behavioral data evolved over time?

## Reproduce the Results and ICR outputs

The RQ scripts are import-safe and execute only through their `main()` entry
points. By default they read the final annotation workbook under `data/` and
write under `output/`. Set `ASD_REVIEW_DATA_ROOT` and
`ASD_REVIEW_OUTPUT_ROOT` to use isolated inputs/outputs.

```bash
python3 scripts/rq1_.py
python3 scripts/rq2_.py
python3 scripts/rq3_.py
python3 scripts/rq4_.py
python3 scripts/rq5_.py
python3 scripts/mapping_across_research_questions.py --overwrite
python3 scripts/run_icr_pipeline.py
python3 scripts/intercoderreliability_paper_selection.py
```

The BERT semantic-category ICR is a separate analysis and is not invoked by
`run_icr_pipeline.py`. Run it independently when that additional analysis is
needed:

```bash
python3 scripts/BERT_icr.py \
  --input-workbook "$ASD_REVIEW_DATA_ROOT/ICR.xlsx" \
  --output-dir "$ASD_REVIEW_OUTPUT_ROOT/bert_icr_results"
```

This command requires the sentence-transformer model named in the script. If it
is not already cached locally, the first run requires model-download access.

The task, algorithm-family, learning, evaluation-metric, and accuracy rules are
defined in `scripts/codebook.py` and shared by Results and ICR. Reliability
calculations use `scripts/reliability.py`. The obsolete
`final_descriptive_statistics.py` workflow has been removed; do not recreate or
run it.

The Results-specific and BERT semantic-category ICR methods remain distinct;
their coefficients must not be presented as though they came from one combined
pipeline.

See `scripts/PRISMA_pipeline/README.md` for the reusable screening pipeline.
