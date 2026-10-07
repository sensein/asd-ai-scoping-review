# asd-ai-scoping-review

This repository contains data processing code for the scoping review paper
"Systematic Scoping Review of AI Applications for Automatic Autism Assessment using Behavioral Data".

## Repository structure

```
data/                   # private input workbooks (not committed)
output/                 # generated results and figures (not committed)
scripts/
  rq1_.py ... rq5_.py   # descriptive statistics, one script per research question
  mapping_across_research_questions.py
  Figures/              # manuscript figures
  PRISMA_pipeline/      # reusable record-screening pipeline (see its README)
  run_icr_pipeline.py, BERT_icr.py, intercoderreliability_paper_selection.py
  codebook.py, helper_functions_.py, columns.py, setup_data_.py,
  analysis_common.py, reliability.py   # shared modules
tests/                  # unit tests, run in CI
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

If you only want pip, `pip install -r requirements.txt` installs the same pinned
versions. The PRISMA PDF extractor also needs the Node dependency in `package.json`
(`npm install`).

## Data

Review workbooks and generated outputs are not in version control. Put these
under `data/`, or point `ASD_REVIEW_DATA_ROOT` at a private directory:

| File | Needed by |
|------|-----------|
| `final_annotation_sheet_.xlsx` (sheet `final_data`, two header rows, 172 studies in rows 3 to 174) | the RQ scripts, `mapping_across_research_questions.py`, `intercoderreliability_paper_selection.py` |
| `ICR.xlsx` (coder-paired blocks, two header rows) | `run_icr_pipeline.py`, `BERT_icr.py` |

`run_icr_pipeline.py` lists `ICR_variable_type_classification.xlsx` in its run log
as present or missing, but it never reads it, so the pipeline runs without it.

Generated results go to `output/`, or to `ASD_REVIEW_OUTPUT_ROOT` if you set it.
Importing the shared modules and running the tests need none of the private
workbooks.

## Running the tests

```bash
python -m pytest tests/
```

Run it from the repository root. GitHub Actions runs the same command on every
pull request and every push to `main`.

## Research questions

The canonical manuscript labels are:

1. **Participants** — What are the characteristics of participants included in AI-based autism prediction studies using behavioral data?
2. **Study design** — How are AI-based autism prediction studies using behavioral data designed, conducted, and reported?
3. **Behaviors** — How is behavioral data conceptualized and used in AI-based autism prediction?
4. **AI techniques** — How are AI and machine learning techniques applied to behavioral data for autism prediction?
5. **Paper writing and publishing trends** — How has the literature on AI-based autism prediction using behavioral data evolved over time?

## Reproduce the results

The RQ scripts are import-safe: they only run through their `main()` entry points.
Run these from the repository root, in this order, because the figures read the
RQ outputs:

```bash
python3 scripts/rq1_.py
python3 scripts/rq2_.py
python3 scripts/rq3_.py
python3 scripts/rq4_.py
python3 scripts/rq5_.py
python3 scripts/mapping_across_research_questions.py --overwrite
python3 scripts/run_icr_pipeline.py
```

The task, algorithm-family, learning, evaluation-metric, and accuracy rules are
defined in `scripts/codebook.py` and shared by the Results scripts and the ICR
pipeline. Reliability calculations use `scripts/reliability.py`. Changing a rule in
`codebook.py` or `helper_functions_.py` can change the manuscript numbers, so rerun
everything and check the figures afterward.

### Inter-coder reliability

`run_icr_pipeline.py` computes Krippendorff's alpha over the predefined
categories. The BERT semantic-category analysis is a separate method and is not run
by it:

```bash
python3 scripts/BERT_icr.py \
  --input-workbook "$ASD_REVIEW_DATA_ROOT/ICR.xlsx" \
  --output-dir "$ASD_REVIEW_OUTPUT_ROOT/bert_icr_results"
```

This needs the `sentence-transformers/all-MiniLM-L6-v2` model. If it is not
cached locally, the first run downloads it (set `HF_HUB_OFFLINE=1` once it is
cached to skip the network check). Report the two sets of coefficients as
separate analyses; they do not come from one combined pipeline.

`intercoderreliability_paper_selection.py` prints the stratified sample of 35
papers (fixed seed, writes no files). That sample has already been double-coded.

### Figures

Run a figure script after the RQ script it depends on. Each writes into
`output/figures/` and honors `ASD_REVIEW_OUTPUT_ROOT`.

| Script | Reads | Writes |
|--------|-------|--------|
| `scripts/Figures/PRISMA_diagram.py` | nothing (counts are hardcoded) | `Figure_1_PRISMA_flow_diagram.{png,pdf}` |
| `scripts/Figures/RQ1_Figure2.py` | `output/rq1_results/` | `Figure_2_complete.{png,pdf,svg}` |
| `scripts/Figures/RQ2_Figure.py` | `output/rq2_results/` | `RQ2_Combined_Figure.{png,pdf}` |
| `scripts/Figures/RQ3_Figure.py` | nothing (counts are hardcoded) | `Figure_RQ3_summary_mapping.{png,pdf}` |
| `scripts/Figures/RQ4_Figure.py` | `output/rq4_results/` | `Figure_RQ4_summary.{png,pdf}` |
| `scripts/Figures/Studies_per_year_line_plot.py` | `output/rq5_results/RQ5_publication_year_exact_summary.csv` | `Publication_year_line_plot.{png,pdf}` |

Two figures do not read any output. The PRISMA counts come from the screening
rounds, which are not in this repository. The RQ3 counts matched the `RQ2_*` and
`RQ3_*` CSVs when the figure was reviewed (October 2026), and the script computes
every percentage from them. If the codebook changes, recheck both by hand.

## Screening pipeline

See `scripts/PRISMA_pipeline/README.md` for the reusable record-screening pipeline.
`scripts/PRISMA_pipeline_Fabio/` holds notebooks from earlier screening experiments.
They are kept for history and are not part of the reproducible pipeline.
