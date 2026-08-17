# PRISMA Pipeline Scripts

These scripts provide a reusable, reproducible PRISMA-style screening pipeline for bibliographic records. Review-specific criteria, date ranges, local input paths, and manual decisions are supplied through configuration files or command-line arguments rather than embedded in the core pipeline code.

## Setup

Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
```

Copy the example criteria file and replace every placeholder term with the terms for your review.

```bash
mkdir -p config
cp scripts/PRISMA_pipeline/review_criteria.example.json config/review_criteria.json
```

The pipeline deliberately fails if this file is missing, a required term group
is empty, or example/placeholder terms remain. It never silently substitutes
the example file.

For isolated or external storage, every stage honors:

```bash
export ASD_REVIEW_DATA_ROOT=/path/to/review-data
export ASD_REVIEW_OUTPUT_ROOT=/path/to/review-output
```

## Expected Inputs

- `data/raw/records.xlsx` or `data/raw/records.csv`: bibliographic records with at least a `title` column. Recommended optional columns are `doi`, `journal`, `year_published`, `authors`, `abstract`, `link`, `language`, `document_type`, `keywords`, and `source_database`.
- `config/review_criteria.json`: review-specific screening term groups, date range, and eligibility options.
- Optional manual workbooks in `data/manual/` for supplemental metadata, manual PDF updates, final removals, or open-access classification.
- Optional PDFs in `output/pdfs/` when full text is supplied manually.

Do not commit proprietary PDFs, private datasets, API caches, reviewer notes, downloaded full texts, extracted full-text files, or generated outputs unless they are explicitly intended to be public.

## Reproduction Order

```bash
python3 scripts/PRISMA_pipeline/00_enrich_metadata.py
python3 scripts/PRISMA_pipeline/01_deduplicate_records.py
python3 scripts/PRISMA_pipeline/02_screen_titles.py --criteria config/review_criteria.json
python3 scripts/PRISMA_pipeline/03_prepare_abstract_screening.py --criteria config/review_criteria.json
python3 scripts/PRISMA_pipeline/04_find_abstracts.py
python3 scripts/PRISMA_pipeline/05_merge_supplemental_metadata.py
python3 scripts/PRISMA_pipeline/06_screen_abstracts.py --criteria config/review_criteria.json
python3 scripts/PRISMA_pipeline/07_apply_manual_pdf_updates.py
python3 scripts/PRISMA_pipeline/08_find_open_access_pdfs.py
python3 scripts/PRISMA_pipeline/10_screen_full_texts.py --criteria config/review_criteria.json
python3 scripts/PRISMA_pipeline/11_create_final_decisions.py
python3 scripts/PRISMA_pipeline/12_apply_manual_removals.py
python3 scripts/PRISMA_pipeline/13_create_final_metadata.py
python3 scripts/PRISMA_pipeline/14_create_open_access_classification.py
```

`10_screen_full_texts.py` calls `09_extract_pdf_texts_pdfjs.mjs` internally. Use `--skip-extraction` only when `output/full_text_screening/pdf_text_extraction_manifest.jsonl` already exists. Under the current review protocol, stage 03 and stage 06 both advance title-stage `Include` records only; title-stage `Maybe` records do not advance. Use `--title-decisions Include,Maybe` only for an explicitly documented future protocol.

## Configuration

The example criteria file is a template, not ASD-specific configuration. Replace every placeholder term group before running a real review. Core decision labels are `Include`, `Maybe`, and `Exclude`; uncertain decisions remain reviewable as `Maybe`.

Networked metadata and PDF discovery stages query public scholarly APIs, use the shared retry/cache behavior in `prisma_common.py`, and write JSON caches under `output/` for rerun auditing. If API access is unavailable, supply equivalent metadata or PDFs through the documented input files and skip the network lookup stages.

## Outputs And PRISMA Counts

Each stage writes an audit workbook or CSV under `output/` with a `Summary` sheet or count table. Deduplication counts come from `Deduplicated_Retained` and duplicate logs. Title, abstract, PDF retrieval, full-text, manual-removal, and final-inclusion counts are derived from the decision columns written by each stage. Manual overrides are supplied through `data/manual/` workbooks and are reflected in separate audit sheets.
