# PRISMA_pipeline_Fabio

Historical record of the original literature search and screening notebooks (Fabio's Colab workflow). These predate the reusable pipeline in `scripts/PRISMA_pipeline/` and are kept for provenance and auditability; they read from a Google Drive workspace and are not part of the reproducible pipeline.

## Workflow order

**1. Per-database export conversion** — convert raw database exports into per-database CSVs with a common schema (`Title`, `Abstract`, `Keywords`, `DOI`, `URL`, `Authors`, `Venue`, `Year`):

- `bib_to_csv.ipynb` — BibTeX exports (e.g. ACM Digital Library) to CSV
- `csv_to_csv.ipynb` — database CSV exports normalized to the common schema
- `pubmed_id_download.ipynb` — fetches PubMed record metadata via Entrez for exported PMID lists

**2. Combination and filtering**

- `0_local_filtering_autism.ipynb` — initial autism-term filtering
- `1_combine_databases.ipynb` — merges the per-database CSVs
- `2_remove_duplicates.ipynb` — deduplication
- `3_local_filtering.ipynb` — keyword-based local filtering
- `screening.ipynb` — screening passes

**3. Experiments and diagnostics**

- `mic.ipynb` — Azure OpenAI GPT-based screening experiment over the query CSVs
- `keyword_stats.ipynb`, `local_filtering_experiments.ipynb`, `experiements.ipynb` — keyword statistics and filtering experiments

## Credentials

API credentials are read from environment variables and must never be hardcoded:

- `NCBI_API_KEY`, `NCBI_EMAIL` — Entrez lookups in `pubmed_id_download.ipynb`
- `AZURE_OPENAI_API_KEY` — `mic.ipynb`
- `OPENAI_API_KEY` — `screening.ipynb`
