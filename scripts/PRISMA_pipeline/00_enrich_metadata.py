#!/usr/bin/env python3
"""Enrich title/journal screening records with DOI, year, authors, abstracts, and links.

The script queries public scholarly metadata APIs and writes an audit-friendly CSV
that can be fed into the downstream screening scripts. It is intentionally conservative:
low-confidence title matches are flagged instead of silently treated as truth.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prisma_common import (
    DATA_ROOT,
    OUTPUT_ROOT,
    load_json_cache,
    normalize_doi as shared_normalize_doi,
    normalize_title as shared_normalize_title,
    request_json_cached,
    save_json_cache,
    title_similarity as shared_title_similarity,
)

DEFAULT_INPUT = DATA_ROOT / "raw" / "records.xlsx"
DEFAULT_OUTPUT = OUTPUT_ROOT / "metadata_enrichment" / "enriched_records.csv"
DEFAULT_CACHE = OUTPUT_ROOT / "metadata_enrichment" / "api_cache.json"
DEFAULT_REVIEW = OUTPUT_ROOT / "metadata_enrichment" / "metadata_review_queue.csv"


def temporary_output_path(output: Path) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}.tmp{output.suffix}")
    return output.with_name(f"{output.name}.tmp")


PIPELINE_COLUMNS = [
    "record_id",
    "title",
    "link",
    "keywords",
    "source_database",
    "year_published",
    "duplicate_status",
    "duplicate_reason",
    "title_screening_decision",
    "title_exclusion_reason",
    "doi",
    "authors",
    "abstract",
    "document_type",
    "language",
    "abstract_screening_decision",
    "abstract_exclusion_reason",
    "full_text_decision",
    "full_text_exclusion_reason",
    "reviewer_notes",
]


ENRICHMENT_COLUMNS = [
    "metadata_status",
    "metadata_source",
    "metadata_match_score",
    "metadata_match_confidence",
    "matched_title",
    "matched_journal",
    "matched_year",
    "openalex_id",
    "semantic_scholar_paper_id",
    "metadata_notes",
]


HEADER_ALIASES = {
    "title": ["title", "article title", "paper title", "name"],
    "source_database": ["source_database", "source database", "journal", "venue", "source", "journal name"],
    "doi": ["doi", "digital object identifier"],
    "year_published": ["year_published", "year published", "year", "publication year"],
    "link": ["link", "url", "article link", "source url", "doi url"],
    "authors": ["authors", "author"],
    "abstract": ["abstract"],
    "document_type": ["document_type", "document type", "publication type", "type"],
    "language": ["language", "lang"],
}


PROVIDERS = ("crossref", "openalex", "semantic_scholar")


@dataclass
class Candidate:
    source: str
    title: str = ""
    doi: str = ""
    year: str = ""
    authors: str = ""
    abstract: str = ""
    journal: str = ""
    link: str = ""
    document_type: str = ""
    language: str = ""
    openalex_id: str = ""
    semantic_scholar_paper_id: str = ""
    score: float = 0.0
    notes: str = ""


def normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("_", " "))


def normalize_title(value: str) -> str:
    return shared_normalize_title(value)


def normalize_doi(value: str) -> str:
    return shared_normalize_doi(value)


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first(value: Any) -> str:
    if isinstance(value, list) and value:
        return clean_text(value[0])
    return clean_text(value)


def title_similarity(left: str, right: str) -> float:
    return shared_title_similarity(left, right)


def journal_bonus(input_journal: str, candidate_journal: str) -> float:
    left = normalize_title(input_journal)
    right = normalize_title(candidate_journal)
    if not left or not right:
        return 0.0
    if left == right:
        return 0.04
    if left in right or right in left:
        return 0.02
    overlap = len(set(left.split()) & set(right.split()))
    return min(0.02, overlap * 0.005)


def confidence(score: float, high_threshold: float, min_score: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= min_score:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def reconstruct_openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words.append((position, word))
    if not words:
        return ""
    return " ".join(word for _, word in sorted(words))


def load_cache(path: Path) -> dict[str, Any]:
    return load_json_cache(path)


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    save_json_cache(path, cache)


def request_json(
    url: str,
    cache: dict[str, Any],
    *,
    user_agent: str,
    delay_seconds: float,
    timeout: int,
    retries: int,
) -> dict[str, Any] | None:
    return request_json_cached(
        url,
        cache,
        user_agent=user_agent,
        delay_seconds=delay_seconds,
        timeout=timeout,
        retries=retries,
    )


def build_url(base: str, params: dict[str, Any]) -> str:
    clean_params = {key: value for key, value in params.items() if value not in ("", None)}
    return f"{base}?{urllib.parse.urlencode(clean_params)}"


def crossref_authors(item: dict[str, Any]) -> str:
    authors = []
    for author in item.get("author", []) or []:
        given = clean_text(author.get("given", ""))
        family = clean_text(author.get("family", ""))
        name = " ".join(part for part in [given, family] if part)
        if name:
            authors.append(name)
    return "; ".join(authors)


def crossref_year(item: dict[str, Any]) -> str:
    for key in ["published-print", "published-online", "published", "issued", "created"]:
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def crossref_candidate(item: dict[str, Any], input_title: str, input_journal: str, source: str) -> Candidate:
    title = first(item.get("title", ""))
    journal = first(item.get("container-title", ""))
    score = min(1.0, title_similarity(input_title, title) + journal_bonus(input_journal, journal))
    doi = normalize_doi(item.get("DOI", ""))
    return Candidate(
        source=source,
        title=title,
        doi=doi,
        year=crossref_year(item),
        authors=crossref_authors(item),
        abstract=clean_text(item.get("abstract", "")),
        journal=journal,
        link=f"https://doi.org/{doi}" if doi else first(item.get("URL", "")),
        document_type=clean_text(item.get("type", "")),
        language=clean_text(item.get("language", "")),
        score=score,
    )


def query_crossref(
    row: dict[str, str],
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> list[Candidate]:
    title = row.get("title", "")
    journal = row.get("source_database", "")
    doi = normalize_doi(row.get("doi", ""))
    params = {"mailto": args.email} if args.email else {}

    if doi:
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
        if params:
            url = build_url(url, params)
        payload = request_json(
            url,
            cache,
            user_agent=args.user_agent,
            delay_seconds=args.delay,
            timeout=args.timeout,
            retries=args.retries,
        )
        item = ((payload or {}).get("message") or {}) if payload else {}
        if item:
            candidate = crossref_candidate(item, title, journal, "crossref_doi")
            candidate.score = max(candidate.score, 0.98)
            return [candidate]
        return []

    url = build_url(
        "https://api.crossref.org/works",
        {
            "query.bibliographic": title,
            "rows": args.rows_per_provider,
            "select": "DOI,title,container-title,published-print,published-online,published,issued,created,author,abstract,type,language,URL,score",
            **params,
        },
    )
    payload = request_json(
        url,
        cache,
        user_agent=args.user_agent,
        delay_seconds=args.delay,
        timeout=args.timeout,
        retries=args.retries,
    )
    items = ((payload or {}).get("message") or {}).get("items", []) if payload else []
    return [crossref_candidate(item, title, journal, "crossref_search") for item in items]


def openalex_authors(item: dict[str, Any]) -> str:
    names = []
    for authorship in item.get("authorships", []) or []:
        name = clean_text((authorship.get("author") or {}).get("display_name", ""))
        if name:
            names.append(name)
    return "; ".join(names)


def openalex_journal(item: dict[str, Any]) -> str:
    primary = item.get("primary_location") or {}
    source = primary.get("source") or {}
    if source.get("display_name"):
        return clean_text(source.get("display_name"))
    host = item.get("host_venue") or {}
    return clean_text(host.get("display_name", ""))


def openalex_link(item: dict[str, Any]) -> str:
    doi = normalize_doi(item.get("doi", ""))
    if doi:
        return f"https://doi.org/{doi}"
    primary = item.get("primary_location") or {}
    if primary.get("landing_page_url"):
        return clean_text(primary.get("landing_page_url"))
    return clean_text(item.get("id", ""))


def openalex_candidate(item: dict[str, Any], input_title: str, input_journal: str, source: str) -> Candidate:
    title = clean_text(item.get("display_name", ""))
    journal = openalex_journal(item)
    score = min(1.0, title_similarity(input_title, title) + journal_bonus(input_journal, journal))
    return Candidate(
        source=source,
        title=title,
        doi=normalize_doi(item.get("doi", "")),
        year=clean_text(item.get("publication_year", "")),
        authors=openalex_authors(item),
        abstract=clean_text(reconstruct_openalex_abstract(item.get("abstract_inverted_index"))),
        journal=journal,
        link=openalex_link(item),
        document_type=clean_text(item.get("type", "")),
        language=clean_text(item.get("language", "")),
        openalex_id=clean_text(item.get("id", "")),
        score=score,
    )


def query_openalex(
    row: dict[str, str],
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> list[Candidate]:
    title = row.get("title", "")
    journal = row.get("source_database", "")
    doi = normalize_doi(row.get("doi", ""))
    params = {"mailto": args.email} if args.email else {}

    if doi:
        url = build_url(
            "https://api.openalex.org/works",
            {
                "filter": f"doi:{doi}",
                "per-page": 1,
                **params,
            },
        )
        payload = request_json(
            url,
            cache,
            user_agent=args.user_agent,
            delay_seconds=args.delay,
            timeout=args.timeout,
            retries=args.retries,
        )
        items = (payload or {}).get("results", []) if payload else []
        if items:
            candidate = openalex_candidate(items[0], title, journal, "openalex_doi")
            candidate.score = max(candidate.score, 0.98)
            return [candidate]
        return []

    url = build_url(
        "https://api.openalex.org/works",
        {
            "search": title,
            "per-page": args.rows_per_provider,
            **params,
        },
    )
    payload = request_json(
        url,
        cache,
        user_agent=args.user_agent,
        delay_seconds=args.delay,
        timeout=args.timeout,
        retries=args.retries,
    )
    items = (payload or {}).get("results", []) if payload else []
    return [openalex_candidate(item, title, journal, "openalex_search") for item in items]


def semantic_authors(item: dict[str, Any]) -> str:
    return "; ".join(clean_text(author.get("name", "")) for author in item.get("authors", []) if author.get("name"))


def semantic_journal(item: dict[str, Any]) -> str:
    publication_venue = item.get("publicationVenue") or {}
    return clean_text(publication_venue.get("name") or item.get("venue", ""))


def semantic_candidate(item: dict[str, Any], input_title: str, input_journal: str, source: str) -> Candidate:
    title = clean_text(item.get("title", ""))
    journal = semantic_journal(item)
    external_ids = item.get("externalIds") or {}
    doi = normalize_doi(external_ids.get("DOI", ""))
    score = min(1.0, title_similarity(input_title, title) + journal_bonus(input_journal, journal))
    return Candidate(
        source=source,
        title=title,
        doi=doi,
        year=clean_text(item.get("year", "")),
        authors=semantic_authors(item),
        abstract=clean_text(item.get("abstract", "")),
        journal=journal,
        link=clean_text(item.get("url", "")) or (f"https://doi.org/{doi}" if doi else ""),
        document_type=clean_text(item.get("publicationTypes", "")),
        semantic_scholar_paper_id=clean_text(item.get("paperId", "")),
        score=score,
    )


def query_semantic_scholar(
    row: dict[str, str],
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> list[Candidate]:
    title = row.get("title", "")
    journal = row.get("source_database", "")
    doi = normalize_doi(row.get("doi", ""))
    fields = "title,year,authors,abstract,externalIds,url,venue,publicationVenue,publicationTypes"

    if doi:
        url = build_url(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi, safe='')}",
            {"fields": fields},
        )
        payload = request_json(
            url,
            cache,
            user_agent=args.user_agent,
            delay_seconds=args.delay,
            timeout=args.timeout,
            retries=args.retries,
        )
        if payload and "paperId" in payload:
            candidate = semantic_candidate(payload, title, journal, "semantic_scholar_doi")
            candidate.score = max(candidate.score, 0.98)
            return [candidate]
        return []

    url = build_url(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {
            "query": title,
            "limit": args.rows_per_provider,
            "fields": fields,
        },
    )
    payload = request_json(
        url,
        cache,
        user_agent=args.user_agent,
        delay_seconds=args.delay,
        timeout=args.timeout,
        retries=args.retries,
    )
    items = (payload or {}).get("data", []) if payload else []
    return [semantic_candidate(item, title, journal, "semantic_scholar_search") for item in items]


def read_input(path: Path, sheet: str | int | None) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise SystemExit("Excel input requires pandas/openpyxl. Export your file as CSV instead.") from exc
        frame = pd.read_excel(path, sheet_name=sheet or 0, dtype=str).fillna("")
        rows = frame.to_dict(orient="records")
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    return [standardize_row(row, index + 1) for index, row in enumerate(rows)]


def standardize_row(raw: dict[str, Any], index: int) -> dict[str, str]:
    normalized = {normalize_header(key): key for key in raw.keys()}
    row = {column: "" for column in PIPELINE_COLUMNS}
    for canonical, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            source_key = normalized.get(normalize_header(alias))
            if source_key is not None:
                row[canonical] = clean_text(raw.get(source_key, ""))
                break
    if not row["record_id"]:
        row["record_id"] = f"REC{index:05d}"
    for column in ENRICHMENT_COLUMNS:
        row[column] = ""
    return row


def merge_candidate(
    row: dict[str, str],
    candidate: Candidate | None,
    args: argparse.Namespace,
) -> dict[str, str]:
    output = dict(row)
    if candidate is None:
        output["metadata_status"] = "not_found"
        output["metadata_match_confidence"] = "none"
        output["metadata_notes"] = "No candidate returned by selected providers."
        return output

    match_confidence = confidence(candidate.score, args.high_score, args.min_score)
    accepted = candidate.score >= args.min_score or args.accept_low_confidence
    output["metadata_status"] = "matched" if accepted else "needs_review"
    output["metadata_source"] = candidate.source
    output["metadata_match_score"] = f"{candidate.score:.3f}"
    output["metadata_match_confidence"] = match_confidence
    output["matched_title"] = candidate.title
    output["matched_journal"] = candidate.journal
    output["matched_year"] = candidate.year
    output["openalex_id"] = candidate.openalex_id
    output["semantic_scholar_paper_id"] = candidate.semantic_scholar_paper_id

    if not accepted:
        output["metadata_notes"] = "Candidate below minimum score; review before using metadata."
        return output

    fill_map = {
        "doi": candidate.doi,
        "year_published": candidate.year,
        "authors": candidate.authors,
        "abstract": candidate.abstract,
        "link": candidate.link,
        "document_type": candidate.document_type,
        "language": candidate.language,
    }
    for column, value in fill_map.items():
        if value and (args.overwrite or not output.get(column)):
            output[column] = value

    notes = []
    if not candidate.abstract:
        notes.append("No abstract available from matched provider.")
    if not candidate.doi:
        notes.append("No DOI available from matched provider.")
    output["metadata_notes"] = " ".join(notes)
    return output


def choose_best_candidate(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            item.score,
            bool(item.abstract),
            bool(item.doi),
            bool(item.authors),
            bool(item.year),
        ),
        reverse=True,
    )[0]


def enrich_row(row: dict[str, str], cache: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    if not row.get("title"):
        output = dict(row)
        output["metadata_status"] = "skipped"
        output["metadata_notes"] = "Missing title."
        return output

    all_candidates: list[Candidate] = []
    if "crossref" in args.providers:
        all_candidates.extend(query_crossref(row, cache, args))
    if "openalex" in args.providers:
        all_candidates.extend(query_openalex(row, cache, args))
    if "semantic_scholar" in args.providers:
        all_candidates.extend(query_semantic_scholar(row, cache, args))
    return merge_candidate(row, choose_best_candidate(all_candidates), args)


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_output_path(path)
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find DOI, publication year, authors, abstracts, and links for title/journal records."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help=f"Input CSV/XLSX. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Enriched output CSV. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--review-output", default=str(DEFAULT_REVIEW), help=f"Low-confidence review CSV. Default: {DEFAULT_REVIEW}")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help=f"API cache JSON. Default: {DEFAULT_CACHE}")
    parser.add_argument("--sheet", default=None, help="Excel sheet name/index to read. Default: first sheet.")
    parser.add_argument("--email", default="", help="Optional email for polite API usage.")
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS), help="Metadata providers to query.")
    parser.add_argument("--rows-per-provider", type=int, default=5, help="Candidates to request per title search. Default: 5.")
    parser.add_argument("--min-score", type=float, default=0.86, help="Minimum title/journal match score to fill metadata. Default: 0.86.")
    parser.add_argument("--high-score", type=float, default=0.94, help="Score treated as high confidence. Default: 0.94.")
    parser.add_argument("--accept-low-confidence", action="store_true", help="Fill fields even when below --min-score; still flags review.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing DOI/year/authors/abstract/link values.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows; useful for testing.")
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between uncached API requests in seconds. Default: 0.15.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds. Default: 30.")
    parser.add_argument("--retries", type=int, default=3, help="Retries for 429/5xx/network errors. Default: 3.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N records. Default: 1.")
    parser.add_argument(
        "--user-agent",
        default="PRISMA metadata enrichment script (mailto optional)",
        help="HTTP User-Agent header.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    rows = read_input(input_path, args.sheet)
    if args.limit:
        rows = rows[: args.limit]
    cache = load_cache(Path(args.cache))

    enriched: list[dict[str, str]] = []
    total = len(rows)
    print(f"Starting metadata enrichment for {total} record(s).", file=sys.stderr, flush=True)
    for index, row in enumerate(rows, start=1):
        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == total):
            title_preview = row.get("title", "").strip()
            if len(title_preview) > 90:
                title_preview = title_preview[:87] + "..."
            print(f"Querying {index}/{total}: {title_preview}", file=sys.stderr, flush=True)
        enriched_row = enrich_row(row, cache, args)
        enriched.append(enriched_row)
        if index % 25 == 0 or index == total:
            save_cache(Path(args.cache), cache)
            print(f"Saved progress after {index}/{total}", file=sys.stderr, flush=True)

    output_columns = PIPELINE_COLUMNS + ENRICHMENT_COLUMNS
    write_csv(Path(args.output), enriched, output_columns)
    review_rows = [
        row for row in enriched
        if row.get("metadata_status") in {"needs_review", "not_found", "skipped"}
        or row.get("metadata_match_confidence") in {"low", "none"}
    ]
    write_csv(Path(args.review_output), review_rows, output_columns)
    save_cache(Path(args.cache), cache)

    matched = sum(1 for row in enriched if row.get("metadata_status") == "matched")
    needs_review = len(review_rows)
    print(f"Wrote enriched records: {Path(args.output)}")
    print(f"Matched records: {matched}/{total}")
    print(f"Review queue: {needs_review}/{total} -> {Path(args.review_output)}")


if __name__ == "__main__":
    main()
