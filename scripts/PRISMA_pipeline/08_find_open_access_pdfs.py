#!/usr/bin/env python3
"""Find publicly available PDFs for final abstract-Include papers.

This script reads the updated abstract-screening workbook, keeps only rows with
final_abstract_screening_decision == Include, searches public/open-access
metadata sources for PDF URLs, downloads valid PDFs into the configured PDF
folder, and writes a retrieval manifest workbook.

It intentionally does not bypass paywalls. A record is marked not_found when no
public PDF URL is exposed by the queried metadata sources.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prisma_common import (
    OUTPUT_ROOT,
    load_json_cache,
    normalize_doi as shared_normalize_doi,
    normalize_title as shared_normalize_title,
    request_text_cached,
    save_json_cache,
    style_workbook as shared_style_workbook,
    title_similarity as shared_title_similarity,
)

import pandas as pd


DEFAULT_INPUT = OUTPUT_ROOT / "abstract_screening" / "title_include_abstract_screening_manual_pdf_updated.xlsx"
DEFAULT_PDF_DIR = OUTPUT_ROOT / "pdfs"
DEFAULT_OUTPUT = OUTPUT_ROOT / "pdf_retrieval" / "pdf_retrieval_manifest.xlsx"
DEFAULT_CACHE = OUTPUT_ROOT / "pdf_retrieval" / "pdf_api_cache.json"


def temporary_output_path(output: Path) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}.tmp{output.suffix}")
    return output.with_name(f"{output.name}.tmp")


@dataclass
class PdfCandidate:
    source: str
    url: str
    title: str = ""
    doi: str = ""
    score: float = 0.0
    notes: str = ""


def clean(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_title(value: Any) -> str:
    return shared_normalize_title(value)


def normalize_doi(value: Any) -> str:
    return shared_normalize_doi(value)


def title_similarity(left: Any, right: Any) -> float:
    return shared_title_similarity(left, right, include_token_jaccard=True)


def safe_filename(record_id: str, title: str) -> str:
    text = normalize_title(title) or "untitled"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return f"{record_id}_{text[:95]}.pdf"


def load_cache(path: Path) -> dict[str, Any]:
    return load_json_cache(path)


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    save_json_cache(path, cache)


def request_text(url: str, cache: dict[str, Any], args: argparse.Namespace, accept: str = "application/json") -> str | None:
    return request_text_cached(
        url, cache, user_agent=args.user_agent, delay_seconds=args.delay,
        timeout=args.timeout, retries=args.retries, accept=accept
    )

def request_json(url: str, cache: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    key = f"JSON::{url}"
    if key in cache:
        value = cache[key]
        return value if isinstance(value, dict) else None
    text = request_text(url, cache, args)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    cache[key] = payload
    return payload


def add_candidate(candidates: list[PdfCandidate], source: str, url: Any, title: str = "", doi: str = "", score: float = 1.0, notes: str = "") -> None:
    cleaned = clean(url)
    if not cleaned or not cleaned.lower().startswith(("http://", "https://")):
        return
    candidates.append(PdfCandidate(source=source, url=cleaned, title=title, doi=doi, score=score, notes=notes))


def openalex_candidates(row: pd.Series, cache: dict[str, Any], args: argparse.Namespace) -> list[PdfCandidate]:
    title = clean(row.get("title", ""))
    doi = normalize_doi(row.get("doi", ""))
    openalex_id = clean(row.get("openalex_id", ""))
    urls: list[str] = []
    if openalex_id:
        work_id = openalex_id.replace("https://openalex.org/", "")
        urls.append(f"https://api.openalex.org/works/{urllib.parse.quote(work_id)}")
    if doi:
        params = urllib.parse.urlencode({"filter": f"doi:{doi}", "per-page": 1})
        urls.append(f"https://api.openalex.org/works?{params}")
    if title:
        params = urllib.parse.urlencode({"search": title, "per-page": 3})
        urls.append(f"https://api.openalex.org/works?{params}")

    candidates: list[PdfCandidate] = []
    seen_work_ids: set[str] = set()
    for url in urls:
        payload = request_json(url, cache, args)
        if not payload:
            continue
        works = payload.get("results") if isinstance(payload.get("results"), list) else [payload]
        for work in works:
            if not isinstance(work, dict):
                continue
            work_id = clean((work.get("ids") or {}).get("openalex", "") or work.get("id", ""))
            if work_id in seen_work_ids:
                continue
            seen_work_ids.add(work_id)
            candidate_title = clean(work.get("display_name", ""))
            score = 1.0 if doi and normalize_doi((work.get("ids") or {}).get("doi", "")) == doi else title_similarity(title, candidate_title)
            if score < args.min_title_score:
                continue
            locations = []
            for key in ("best_oa_location", "primary_location"):
                if isinstance(work.get(key), dict):
                    locations.append(work[key])
            locations.extend(work.get("locations") or [])
            for loc in locations:
                if not isinstance(loc, dict):
                    continue
                add_candidate(candidates, "openalex_pdf_url", loc.get("pdf_url", ""), candidate_title, doi, score, "OpenAlex location pdf_url.")
                add_candidate(candidates, "openalex_landing_page", loc.get("landing_page_url", ""), candidate_title, doi, score, "OpenAlex landing_page_url candidate.")
            oa = work.get("open_access") or {}
            add_candidate(candidates, "openalex_oa_url", oa.get("oa_url", ""), candidate_title, doi, score, "OpenAlex open_access oa_url candidate.")
    return candidates


def europe_pmc_candidates(row: pd.Series, cache: dict[str, Any], args: argparse.Namespace) -> list[PdfCandidate]:
    title = clean(row.get("title", ""))
    doi = normalize_doi(row.get("doi", ""))
    queries = []
    if doi:
        queries.append(f'DOI:"{doi}"')
    if title:
        queries.append(f'TITLE:"{title.replace(chr(34), " ")}"')
    candidates: list[PdfCandidate] = []
    for query in queries:
        params = urllib.parse.urlencode({"query": query, "format": "json", "pageSize": 3, "resultType": "core"})
        payload = request_json(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}", cache, args)
        results = (((payload or {}).get("resultList") or {}).get("result")) or []
        for result in results:
            candidate_title = clean(result.get("title", ""))
            score = 1.0 if doi and normalize_doi(result.get("doi", "")) == doi else title_similarity(title, candidate_title)
            if score < args.min_title_score:
                continue
            url_list = ((result.get("fullTextUrlList") or {}).get("fullTextUrl")) or []
            for item in url_list:
                if not isinstance(item, dict):
                    continue
                style = clean(item.get("documentStyle", "")).lower()
                availability = clean(item.get("availability", "")).lower()
                if "pdf" in style or item.get("url", "").lower().endswith(".pdf") or "open access" in availability:
                    add_candidate(candidates, "europe_pmc_fulltext", item.get("url", ""), candidate_title, doi, score, f"Europe PMC {style} {availability}.")
            pmcid = clean(result.get("pmcid", ""))
            if pmcid:
                add_candidate(candidates, "europe_pmc_pmc_pdf", f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/", candidate_title, doi, score, "Derived PMC PDF URL.")
    return candidates


def semantic_scholar_candidates(row: pd.Series, cache: dict[str, Any], args: argparse.Namespace) -> list[PdfCandidate]:
    title = clean(row.get("title", ""))
    doi = normalize_doi(row.get("doi", ""))
    fields = "title,year,url,externalIds,openAccessPdf"
    urls = []
    if doi:
        urls.append(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields={fields}")
    if title:
        params = urllib.parse.urlencode({"query": title, "limit": 3, "fields": fields})
        urls.append(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}")
    candidates: list[PdfCandidate] = []
    for url in urls:
        payload = request_json(url, cache, args)
        if not payload:
            continue
        papers = payload.get("data") if isinstance(payload.get("data"), list) else [payload]
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            candidate_title = clean(paper.get("title", ""))
            external = paper.get("externalIds") or {}
            score = 1.0 if doi and normalize_doi(external.get("DOI", "")) == doi else title_similarity(title, candidate_title)
            if score < args.min_title_score:
                continue
            oa_pdf = paper.get("openAccessPdf") or {}
            add_candidate(candidates, "semantic_scholar_open_pdf", oa_pdf.get("url", ""), candidate_title, doi, score, "Semantic Scholar openAccessPdf.url.")
    return candidates


def crossref_candidates(row: pd.Series, cache: dict[str, Any], args: argparse.Namespace) -> list[PdfCandidate]:
    title = clean(row.get("title", ""))
    doi = normalize_doi(row.get("doi", ""))
    urls = []
    if doi:
        urls.append(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    elif title:
        params = urllib.parse.urlencode({"query.bibliographic": title, "rows": 3})
        urls.append(f"https://api.crossref.org/works?{params}")
    candidates: list[PdfCandidate] = []
    for url in urls:
        payload = request_json(url, cache, args)
        if not payload:
            continue
        message = payload.get("message") or {}
        items = message.get("items") if isinstance(message.get("items"), list) else [message]
        for item in items:
            candidate_title = clean((item.get("title") or [""])[0] if isinstance(item.get("title"), list) else item.get("title", ""))
            score = 1.0 if doi and normalize_doi(item.get("DOI", "")) == doi else title_similarity(title, candidate_title)
            if score < args.min_title_score:
                continue
            for link in item.get("link") or []:
                if not isinstance(link, dict):
                    continue
                content_type = clean(link.get("content-type", "")).lower()
                if "pdf" in content_type or clean(link.get("URL", "")).lower().endswith(".pdf"):
                    add_candidate(candidates, "crossref_pdf_link", link.get("URL", ""), candidate_title, doi, score, f"Crossref link content-type {content_type}.")
    return candidates


def direct_candidates(row: pd.Series) -> list[PdfCandidate]:
    candidates: list[PdfCandidate] = []
    title = clean(row.get("title", ""))
    doi = normalize_doi(row.get("doi", ""))
    for column in ("link", "abstract_lookup_matched_url"):
        url = clean(row.get(column, ""))
        if ".pdf" in url.lower() or "/pdf" in url.lower():
            add_candidate(candidates, f"direct_{column}", url, title, doi, 1.0, f"Direct metadata URL from {column}.")
    return candidates


def provider_candidates(row: pd.Series, cache: dict[str, Any], args: argparse.Namespace) -> list[PdfCandidate]:
    providers = [item.strip().lower() for item in args.providers.split(",") if item.strip()]
    candidates = direct_candidates(row)
    if "europe_pmc" in providers:
        candidates.extend(europe_pmc_candidates(row, cache, args))
    if "openalex" in providers:
        candidates.extend(openalex_candidates(row, cache, args))
    if "semantic_scholar" in providers:
        candidates.extend(semantic_scholar_candidates(row, cache, args))
    if "crossref" in providers:
        candidates.extend(crossref_candidates(row, cache, args))
    deduped: list[PdfCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.url.split("#")[0]
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def existing_pdf_match(row: pd.Series, pdf_dir: Path, min_score: float) -> Path | None:
    title = clean(row.get("title", ""))
    best_score = 0.0
    best_path: Path | None = None
    for path in pdf_dir.glob("*.pdf"):
        score = title_similarity(path.stem, title)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path if best_path and best_score >= min_score else None


def download_pdf(url: str, destination: Path, args: argparse.Namespace) -> tuple[bool, str, int, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": args.user_agent,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.2",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=args.download_timeout) as response:
            final_url = response.geturl()
            content_type = clean(response.headers.get("Content-Type", "")).lower()
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > args.max_pdf_mb * 1024 * 1024:
                    return False, f"PDF exceeded {args.max_pdf_mb} MB limit", total, final_url
            data = b"".join(chunks)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}", 0, url
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, f"Download error: {exc}", 0, url

    if not data.startswith(b"%PDF"):
        if "pdf" not in content_type:
            return False, f"Not a PDF; content-type={content_type or 'unknown'}", len(data), final_url
        if b"%PDF" not in data[:2048]:
            return False, f"PDF header not found; content-type={content_type}", len(data), final_url
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = temporary_output_path(destination)
    temp_destination.write_bytes(data)
    temp_destination.replace(destination)
    return True, "downloaded", len(data), final_url


def read_skip_record_ids(path: Path | None) -> set[str]:
    if not path:
        return set()
    if not path.exists():
        raise SystemExit(f"Skip manifest not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path)
        sheet = "PDF_Retrieval_Manifest" if "PDF_Retrieval_Manifest" in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")
    else:
        df = pd.read_csv(path, dtype=str).fillna("")
    if "record_id" not in df.columns:
        raise SystemExit(f"Skip manifest is missing record_id column: {path}")
    return set(df["record_id"].astype(str).str.strip())


def read_targets(path: Path, sheet: str, limit: int, decisions: str, skip_manifest: Path | None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")
    if "final_abstract_screening_decision" not in df.columns:
        raise SystemExit("Input sheet is missing final_abstract_screening_decision.")
    wanted = {item.strip() for item in decisions.split(",") if item.strip()}
    if not wanted:
        raise SystemExit("No decisions requested.")
    targets = df[df["final_abstract_screening_decision"].isin(wanted)].copy()
    skip_ids = read_skip_record_ids(skip_manifest)
    if skip_ids:
        targets = targets[~targets["record_id"].astype(str).str.strip().isin(skip_ids)].copy()
    if limit:
        targets = targets.head(limit).copy()
    return targets


def style_workbook(path: Path) -> None:
    shared_style_workbook(path)

def write_manifest(output: Path, manifest: pd.DataFrame, target_count: int, pdf_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    status_counts = manifest["pdf_retrieval_status"].value_counts(dropna=False).rename_axis("pdf_retrieval_status").reset_index(name="count")
    source_counts = manifest.loc[manifest["pdf_source"].astype(str).str.strip().ne(""), "pdf_source"].value_counts(dropna=False).rename_axis("pdf_source").reset_index(name="count")
    found_mask = manifest["pdf_retrieval_status"].isin(["downloaded", "already_existing"])
    summary = pd.DataFrame(
        [
            {"metric": "Target records in this run", "value": target_count},
            {"metric": "PDF folder", "value": str(pdf_dir.resolve())},
            {"metric": "PDFs available after retrieval", "value": int(found_mask.sum())},
            {"metric": "New PDFs downloaded", "value": int((manifest["pdf_retrieval_status"] == "downloaded").sum())},
            {"metric": "Already existing PDFs matched", "value": int((manifest["pdf_retrieval_status"] == "already_existing").sum())},
            {"metric": "PDFs not found", "value": int((manifest["pdf_retrieval_status"] == "not_found").sum())},
        ]
    )
    temp_output = temporary_output_path(output)
    with pd.ExcelWriter(temp_output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        status_counts.to_excel(writer, sheet_name="Status_Counts", index=False)
        source_counts.to_excel(writer, sheet_name="Source_Counts", index=False)
        manifest.to_excel(writer, sheet_name="PDF_Retrieval_Manifest", index=False)
        manifest.loc[found_mask].to_excel(writer, sheet_name="PDFs_Found", index=False)
        manifest.loc[~found_mask].to_excel(writer, sheet_name="PDFs_Not_Found", index=False)
    style_workbook(temp_output)
    temp_output.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find public PDFs for final abstract-Include records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sheet", default="Title_Include_Screening")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--providers", default="europe_pmc,openalex,semantic_scholar,crossref")
    parser.add_argument("--decisions", default="Include,Maybe", help="Comma-separated final abstract decisions to retrieve PDFs for, e.g. Include,Maybe.")
    parser.add_argument("--skip-manifest", type=Path, default=None, help="Optional prior PDF_Retrieval_Manifest workbook/csv whose record_id values should be skipped.")
    parser.add_argument("--min-title-score", type=float, default=0.88)
    parser.add_argument("--existing-match-score", type=float, default=0.94)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--timeout", type=int, default=18)
    parser.add_argument("--download-timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-pdf-mb", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--user-agent", default="prisma-open-pdf-finder/1.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input workbook not found: {args.input}", file=sys.stderr)
        return 1
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache)
    targets = read_targets(args.input, args.sheet, args.limit, args.decisions, args.skip_manifest)
    manifest_rows: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(targets.iterrows(), start=1):
        record_id = clean(row.get("record_id", "")) or f"ROW{position:04d}"
        title = clean(row.get("title", ""))
        doi = normalize_doi(row.get("doi", ""))
        existing = existing_pdf_match(row, args.pdf_dir, args.existing_match_score)
        base = {
            "record_id": record_id,
            "title": title,
            "doi": doi,
            "link": clean(row.get("link", "")),
            "year_published": clean(row.get("year_published", "")),
            "journal": clean(row.get("journal", "") or row.get("source_database", "")),
        }

        if existing:
            manifest_rows.append(
                {
                    **base,
                    "pdf_retrieval_status": "already_existing",
                    "pdf_source": "existing_pdf_folder",
                    "pdf_url": "",
                    "pdf_final_url": "",
                    "pdf_path": str(existing.resolve()),
                    "pdf_bytes": existing.stat().st_size,
                    "pdf_attempts": 0,
                    "pdf_notes": "Matched an existing PDF in the folder by title.",
                }
            )
            continue

        candidates = provider_candidates(row, cache, args)
        downloaded = False
        attempt_notes: list[str] = []
        for candidate in sorted(candidates, key=lambda item: (item.source.startswith("europe"), item.score), reverse=True):
            filename = safe_filename(record_id, title)
            destination = args.pdf_dir / filename
            if destination.exists():
                digest = hashlib.sha1(candidate.url.encode("utf-8")).hexdigest()[:8]
                destination = args.pdf_dir / f"{destination.stem}_{digest}.pdf"
            ok, note, byte_count, final_url = download_pdf(candidate.url, destination, args)
            attempt_notes.append(f"{candidate.source}: {note}")
            if ok:
                manifest_rows.append(
                    {
                        **base,
                        "pdf_retrieval_status": "downloaded",
                        "pdf_source": candidate.source,
                        "pdf_url": candidate.url,
                        "pdf_final_url": final_url,
                        "pdf_path": str(destination.resolve()),
                        "pdf_bytes": byte_count,
                        "pdf_attempts": len(attempt_notes),
                        "pdf_notes": candidate.notes,
                    }
                )
                downloaded = True
                break

        if not downloaded:
            manifest_rows.append(
                {
                    **base,
                    "pdf_retrieval_status": "not_found",
                    "pdf_source": "",
                    "pdf_url": "",
                    "pdf_final_url": "",
                    "pdf_path": "",
                    "pdf_bytes": "",
                    "pdf_attempts": len(candidates),
                    "pdf_notes": "; ".join(attempt_notes[:8]) if attempt_notes else "No public PDF URL found in queried metadata sources.",
                }
            )

        if position % args.save_every == 0:
            save_cache(args.cache, cache)
        if position % args.progress_every == 0 or position == len(targets):
            found = sum(1 for item in manifest_rows if item["pdf_retrieval_status"] in {"downloaded", "already_existing"})
            downloaded_count = sum(1 for item in manifest_rows if item["pdf_retrieval_status"] == "downloaded")
            print(f"Processed {position}/{len(targets)} | PDFs available {found} | new downloads {downloaded_count}", flush=True)

    save_cache(args.cache, cache)
    manifest = pd.DataFrame(manifest_rows)
    write_manifest(args.output, manifest, len(targets), args.pdf_dir)
    print(f"Wrote manifest: {args.output}", flush=True)
    print(manifest["pdf_retrieval_status"].value_counts(dropna=False).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
