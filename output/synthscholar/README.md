# SynthScholar review outputs

Machine-generated systematic-review artifacts for the corpus of included studies in
*"Systematic Scoping Review of AI Applications for Automatic Autism Assessment using
Behavioral Data"*, produced with [SynthScholar](https://github.com/sensein/synthscholar)
via the bring-your-own-corpus workflow of the
[synthscholar agent skill](https://github.com/sensein/agent_skills/tree/main/skills/synthscholar).

## How these files were generated

- **Run date:** 2026-08-15 (`review.json` timestamp `2026-08-15T16:31:12Z`).
- **Corpus:** 171 human-screened PDFs supplied from the project's shared Google Drive
  folder; 1 duplicate removed (Li, Mache & Todd 2020 appears twice in the folder);
  **170 studies included**, all assessed on full text (`full_text_source: user_supplied_pdf`).
- **Protocol:** `protocol.json` (extracted verbatim from `review.json`; title
  *"Computation-based autism prediction from behavioral data: a systematic review of
  study design, conduct and reporting"*), with 23 prespecified research questions
  charted per study and QUADAS-2 risk-of-bias assessment. `max_hops = 0`
  (no citation chasing — the corpus is closed).
- **Copyright:** extracted publisher full text was removed before export
  (`full_text_withheld`); each article retains its `content_sha256` so a
  reconstruction from the source PDFs can be verified.

## Files

| File | Contents |
|------|----------|
| `review.json` | Source of truth — the full `PRISMAReviewResult` (all other files derive from it) |
| `review.md` | PRISMA 2020 review document |
| `review.ttl` | SLR-ontology RDF export (triple-store ready) |
| `review.research-questions.md` | Per-study answers to the 23 prespecified research questions |
| `review.appraisal.{json,md}` | Critical-appraisal tables |
| `review.narrative.{json,md}` | Condensed narrative synthesis |
| `review.per-group.md` | Per-group synthesis and Q&A |
| `protocol.json` | Review protocol (extracted from `review.json`) |
| `corpus_manifest.csv` | 170 included studies: pmid, title, year, journal, DOI, `content_sha256`, source Google Drive file id + citation filename, and which metadata fields were patched |
| `review.bib` | Generated BibTeX for the 170 included studies |

## Post-run metadata repairs (2026-08-18)

`build_corpus.py` guesses title/DOI metadata from PDF text; the raw run left gaps that
were repaired against the human-curated citation filenames of the source Drive folder
(each file is named with its full APA citation). Applied consistently to `review.json`,
`review.md`, `review.research-questions.md`, `review.appraisal.md`, and `review.ttl`
(verified with rdflib: only `bibo:doi`, `dcterms:title`, and
`dcterms:bibliographicCitation` triples changed):

- **88 missing DOIs filled** and **2 malformed DOIs corrected** (a publisher template
  placeholder and a truncated value). All 90 verified to resolve against the
  `doi.org` handle registry.
- **12 junk titles repaired** (PDF header artifacts such as "Contents lists available
  at ScienceDirect", "Original Investigation | Psychiatry", "Microsoft Word - …"),
  with the matching journal fields.
- 2 studies legitimately have no DOI (ACL Anthology / ACM IUI workshop papers):
  `local_afdb603d` (Beccaria et al. 2022), `local_6249bd63` (Sariyanidi et al. 2023).
- Per-study patch provenance is recorded in the `metadata_patched` column of
  `corpus_manifest.csv`.

Untouched known imperfection: many `journal` strings are truncated PDF-metadata
guesses (e.g. "Journal of Autism and Develo"); only junk/empty ones tied to the
repaired studies were fixed. A full journal-field clean would need a corpus rebuild
with a metadata manifest.

## Known gaps

- **Search provenance is deliberately undeclared** — `review.md` records
  *"Not yet declared by the reviewer — user-supplied corpus"*. The databases, query
  strings, dates, and pre-screening record counts of the original PRISMA screening
  are not in this repository. When available, patch them in **without re-running the
  analysis** via the skill: `python update_provenance.py review.json --provenance
  search_provenance.json --outdir .`
- The LLM model used for the run is not recorded in `review.json`.
- Re-exporting these files from `review.json` requires the **development checkout**
  of the SynthScholar engine (`prisma-review-agent`); public releases (PyPI ≤ 0.0.11,
  GitHub `sensein/synthscholar` main) predate the provenance schema and would
  silently drop fields (Pydantic discards unknown keys). The metadata repairs above
  were therefore applied as verified surgical edits, not a re-export.
