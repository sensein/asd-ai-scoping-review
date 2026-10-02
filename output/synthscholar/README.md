# SynthScholar review outputs

These files are supporting material for the paper "Systematic Scoping Review of AI
Applications for Automatic Autism Assessment using Behavioral Data". They were generated
with [SynthScholar](https://github.com/sensein/synthscholar), using the bring-your-own-corpus
workflow of the
[synthscholar agent skill](https://github.com/sensein/agent_skills/tree/main/skills/synthscholar).

## Read this first

The run only analysed papers we had already found and screened. It did not search or
screen anything. The search and screening are described in the paper and its PRISMA flow
diagram (111 included studies from 2013-2023, 61 from 2024-2026, 172 in total). Some
sentences in `review.md`, such as "A systematic search of Not yet declared by the
reviewer - user-supplied corpus was conducted", come from a template with an empty search
field. They do not describe a search.

An LLM agent wrote the per-study charting, appraisal and synthesis, and nobody has
validated it independently. Check any study-level detail against the paper or the source
article before you rely on it.

The grounding scores mean nothing. All 1,580 evidence spans in `review.json` have
`grounded: true` and `grounding_score: 1.0`, and `grounding_validation` is null. Those
are default values, not the result of a check.

Evidence spans are short quotes (median length about 200 characters), and each article
includes its abstract. Full text is not included.

## How this corpus differs from the paper

The corpus has 170 studies and the paper analyses 172. Matching on DOI and title (DOIs
checked against Crossref), 169 studies are in both.

Three studies in the paper are not in this run:

| Study | Venue, year | DOI |
|-------|-------------|-----|
| Li, Zhong, Han, Ouyang, Li & Liu, *Classifying ASD children with LSTM based on raw videos* | Neurocomputing, 2020 | 10.1016/j.neucom.2019.05.106 |
| Che Hasan, Jailani & Md Tahir, *Use of statistical approaches and artificial neural networks to identify gait deviations in children with autism spectrum disorder* | Int. J. Biology and Biomedical Engineering, 2017 | none |
| Alghamdi & Dardouri, *Early Autism Spectrum Disorder Detection Using SMOTE-Enhanced 1D-CNN and Behavioral Screening Data* | Advances in Human-Computer Interaction, 2025 | 10.1155/ahci/1316854 |

One study in this run is not in the paper: *Performance Evaluation of Neural Network
Models for Autism Detection Using EEG Data* (Advances in Technology Innovation, 2024, DOI
10.46604/aiti.2024.13951, id `local_0b0960c4`). Its title in `review.json` still has
"[REC04023]" from the PDF.

Because of this, counts you derive from these files will not exactly match the paper. Use
the paper's numbers.

## How the files were generated

- Run date: 2026-08-15 (`review.json` timestamp `2026-08-15T16:31:12Z`).
- Corpus: 171 PDFs from the project's shared Google Drive folder. One was a duplicate
  (Li, Mache & Todd 2020 appears twice), which left 170 studies, all read as full text
  (`full_text_source: user_supplied_pdf`).
- Protocol: `protocol.json`, taken unchanged from `review.json`. The title is
  "Computation-based autism prediction from behavioral data: a systematic review of study
  design, conduct and reporting". It has 23 research questions charted per study and
  QUADAS-2 for risk of bias. `max_hops = 0`, so there was no citation chasing.
- Copyright: the extracted publisher full text was removed before export
  (`full_text_withheld`). Each article keeps its `content_sha256`, so you can check a
  rebuild from the source PDFs against it.

## Files

| File | Contents |
|------|----------|
| `review.json` | The full `PRISMAReviewResult`. All other files come from this one. |
| `review.md` | The PRISMA 2020 review document |
| `review.ttl` | RDF export using the SLR ontology |
| `review.research-questions.md` | Per-study answers to the 23 research questions |
| `review.appraisal.{json,md}` | Critical-appraisal tables |
| `review.narrative.{json,md}` | Short narrative synthesis |
| `review.per-group.md` | Per-group synthesis and Q&A |
| `protocol.json` | The review protocol, extracted from `review.json` |
| `corpus_manifest.csv` | The 170 studies: pmid, title, year, journal, DOI, `content_sha256`, Google Drive file id and citation filename, and which metadata fields were patched |
| `review.bib` | BibTeX for the 170 studies |

## Metadata repairs after the run (2026-08-18)

`build_corpus.py` guesses titles and DOIs from the PDF text, and the raw run had gaps. We
repaired them using the citation filenames in the Drive folder, which are full APA
citations written by hand. The changes were applied to `review.json`, `review.md`,
`review.research-questions.md`, `review.appraisal.md` and `review.ttl`. In the `.ttl`, we
checked with rdflib that only the `bibo:doi`, `dcterms:title` and
`dcterms:bibliographicCitation` triples changed.

- 88 missing DOIs were filled and 2 bad ones corrected (a publisher template placeholder
  and a truncated value). All 90 resolve at doi.org.
- 12 junk titles were replaced, with their journal fields. The junk was PDF header text
  such as "Contents lists available at ScienceDirect", "Original Investigation |
  Psychiatry" and "Microsoft Word - ...".
- Two studies have no DOI because they are ACL Anthology and ACM IUI workshop papers:
  `local_afdb603d` (Beccaria et al. 2022) and `local_6249bd63` (Sariyanidi et al. 2023).
- The `metadata_patched` column of `corpus_manifest.csv` records what was patched for each
  study.

Many `journal` values are still cut off, for example "Journal of Autism and Develo". We
only fixed the ones tied to the repaired studies. Fixing the rest would mean rebuilding
the corpus with a metadata manifest.

## Known gaps

- Search details (databases, queries, dates, screening counts) are not in these files
  because the run started from an already-screened corpus. They are in the paper. If we
  ever want them in here, `update_provenance.py` in the skill can add them without
  rerunning the analysis: `python update_provenance.py review.json --provenance
  search_provenance.json --outdir .`
- The model and version used are not recorded (`run_configuration` is null in
  `review.json`). The person who ran it says they used the synthscholar agent skill under
  Claude Code, not the SynthScholar application, which is why the application's more
  detailed provenance is missing.
- Grounding validation was not run (see "Read this first").
- The corpus is missing 3 of the paper's studies and has 1 extra (see "How this corpus
  differs from the paper").
- Regenerating these files from `review.json` needs the development checkout of the
  SynthScholar engine (`prisma-review-agent`). The public releases (PyPI up to 0.0.11,
  and GitHub `sensein/synthscholar` main) are older than the provenance schema and would
  silently drop fields, because Pydantic discards keys it does not know. That is why the
  repairs above were careful manual edits and not a re-export.
