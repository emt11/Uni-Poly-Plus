"""Online/local literature retrieval, parsing, and candidate chunk recall."""

"""Lightweight online literature retrieval for the Polymer KG pipeline.

Metadata-first retrieval only. Full text downloads are restricted to open-access
URLs surfaced by public metadata APIs; otherwise a metadata-only .txt document is
written for the existing parser/recall/extraction chain.
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .io_utils import assert_no_val_fields, normalize_alias, read_csv, read_jsonl, stable_id, write_json, write_jsonl


def _progress(iterable, total=None, desc="progress", unit="item"):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return iterable


POLYMER_KEYWORDS = [
    "polymer", "copolymer", "polymerization", "synthesis", "molecular weight",
    "Mn", "Mw", "dispersity", "PDI", "composition", "sequence", "architecture",
    "crosslinked", "network", "characterization",
]
SOURCE_BONUS = {
    "openalex": 0.06,
    "crossref": 0.05,
    "europe_pmc": 0.06,
    "arxiv": 0.04,
    "semantic_scholar": 0.04,
}


def _log(message):
    print(f"[retrieval] {message}", flush=True)


def _request_json(url, timeout=20, headers=None):
    request = urllib.request.Request(url, headers=headers or {"User-Agent": "Uni-Poly-KG/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _strip_html(value):
    return re.sub(r"<[^>]+>", " ", str(value or "")).replace("\n", " ").strip()


def _year(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_filename(value, suffix=".txt"):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120].strip("_") or "article"
    return f"{text}{suffix}"


def _mapping_terms(row):
    terms = []
    for value in row.get("aliases") or []:
        terms.append((str(value), "alias"))
    for field in ("raw_smiles", "canonical_smiles"):
        value = row.get(field)
        if value:
            terms.append((str(value), field))
    seen = set()
    output = []
    for term, source in terms:
        key = (normalize_alias(term), source)
        if key[0] and key not in seen:
            seen.add(key)
            output.append((term, source))
    return output


def build_retrieval_queries(repeat_units_path, mapping_candidates_path, max_repeat_units=None, max_queries_per_repeat_unit=5):
    del repeat_units_path
    queries = []
    seen_queries = set()
    for row in read_jsonl(mapping_candidates_path)[:max_repeat_units]:
        terms = _mapping_terms(row)
        high_value = [(term, source) for term, source in terms if source == "alias"]
        selected = high_value[:max_queries_per_repeat_unit]
        if not selected and row.get("canonical_smiles"):
            selected = [(row["canonical_smiles"], "canonical_smiles")]
        for term, source in selected:
            query_terms = [term]
            query_terms += ["polymer", "synthesis", "molecular weight"]
            query = " ".join(query_terms)
            key = (row.get("repeat_unit_id"), query.lower())
            if key in seen_queries:
                continue
            seen_queries.add(key)
            payload = {
                "query_id": stable_id("query", row.get("repeat_unit_id"), query, length=20),
                "repeat_unit_id": row.get("repeat_unit_id"),
                "query": query,
                "query_terms": query_terms,
                "source_fields": [source],
                "mapping_source_type": row.get("source_type"),
                "mapping_confidence": float(row.get("confidence") or 0),
            }
            assert_no_val_fields(payload, "retrieval_query")
            queries.append(payload)
    return queries


def _normalize_article(source, item, query):
    doi = item.get("doi") or item.get("DOI") or ""
    title = _strip_html(item.get("title") or "")
    abstract = _strip_html(item.get("abstract") or "")
    url = item.get("url") or ""
    full_text_url = item.get("full_text_url") or ""
    oa_status = item.get("oa_status") or ("open" if full_text_url else "unknown")
    article_id = stable_id("art", doi.lower() or title.lower() or url or item.get("source_id"), length=20)
    return {
        "article_id": article_id,
        "title": title,
        "doi": doi,
        "abstract": abstract,
        "year": _year(item.get("year")),
        "venue": item.get("venue") or "",
        "source": source,
        "source_id": item.get("source_id") or "",
        "url": url,
        "oa_status": oa_status,
        "license": item.get("license") or "",
        "full_text_url": full_text_url,
        "full_text_available": bool(full_text_url),
        "matched_repeat_unit_ids": [query["repeat_unit_id"]],
        "matched_queries": [query["query_id"]],
    }


def _search_crossref(query, max_results, timeout):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": query["query"], "rows": max_results})
    data = _request_json(url, timeout=timeout)
    rows = []
    for item in data.get("message", {}).get("items", []):
        links = item.get("link") or []
        full = next((link.get("URL") for link in links if "pdf" in str(link.get("content-type", "")).lower() or "text/html" in str(link.get("content-type", "")).lower()), "")
        rows.append({
            "doi": item.get("DOI", ""),
            "title": " ".join(item.get("title") or []),
            "abstract": item.get("abstract", ""),
            "year": ((item.get("published-print") or item.get("published-online") or item.get("created") or {}).get("date-parts") or [[None]])[0][0],
            "venue": " ".join(item.get("container-title") or []),
            "url": item.get("URL", ""),
            "full_text_url": full if item.get("license") else "",
            "oa_status": "open" if item.get("license") and full else "unknown",
            "license": (item.get("license") or [{}])[0].get("URL", "") if item.get("license") else "",
            "source_id": item.get("DOI", ""),
        })
    return rows


def _search_openalex(query, max_results, timeout):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": query["query"], "per-page": max_results})
    data = _request_json(url, timeout=timeout)
    rows = []
    for item in data.get("results", []):
        best = item.get("best_oa_location") or {}
        rows.append({
            "doi": str(item.get("doi") or "").replace("https://doi.org/", ""),
            "title": item.get("title") or "",
            "abstract": " ".join((item.get("abstract_inverted_index") or {}).keys()),
            "year": item.get("publication_year"),
            "venue": ((item.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
            "url": item.get("id", ""),
            "full_text_url": best.get("pdf_url") or best.get("landing_page_url") or "",
            "oa_status": "open" if item.get("open_access", {}).get("is_oa") else "closed",
            "license": best.get("license") or "",
            "source_id": item.get("id", ""),
        })
    return rows


def _search_europe_pmc(query, max_results, timeout):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({"query": query["query"], "format": "json", "pageSize": max_results})
    data = _request_json(url, timeout=timeout)
    rows = []
    for item in data.get("resultList", {}).get("result", []):
        pmcid = item.get("pmcid") or ""
        full = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML" if pmcid else ""
        rows.append({
            "doi": item.get("doi", ""),
            "title": item.get("title", ""),
            "abstract": item.get("abstractText", ""),
            "year": item.get("pubYear"),
            "venue": item.get("journalTitle", ""),
            "url": item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", "") if item.get("fullTextUrlList") else "",
            "full_text_url": full,
            "oa_status": "open" if pmcid else "unknown",
            "license": item.get("license", ""),
            "source_id": pmcid or item.get("id", ""),
        })
    return rows


def _search_arxiv(query, max_results, timeout):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": "all:" + query["query"], "start": 0, "max_results": max_results})
    request = urllib.request.Request(url, headers={"User-Agent": "Uni-Poly-KG/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    rows = []
    for entry in re.findall(r"<entry>(.*?)</entry>", text, flags=re.S):
        arxiv_id = re.search(r"<id>(.*?)</id>", entry, flags=re.S)
        title = re.search(r"<title>(.*?)</title>", entry, flags=re.S)
        summary = re.search(r"<summary>(.*?)</summary>", entry, flags=re.S)
        year = re.search(r"<published>(\d{4})", entry)
        source_id = arxiv_id.group(1).rsplit("/", 1)[-1] if arxiv_id else ""
        rows.append({
            "doi": "",
            "title": _strip_html(title.group(1) if title else ""),
            "abstract": _strip_html(summary.group(1) if summary else ""),
            "year": year.group(1) if year else None,
            "venue": "arXiv",
            "url": arxiv_id.group(1) if arxiv_id else "",
            "full_text_url": f"https://arxiv.org/pdf/{source_id}.pdf" if source_id else "",
            "oa_status": "open" if source_id else "unknown",
            "license": "arXiv",
            "source_id": source_id,
        })
    return rows


def _search_semantic_scholar(query, max_results, timeout):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode({"query": query["query"], "limit": max_results, "fields": "title,abstract,year,venue,doi,url,openAccessPdf"})
    data = _request_json(url, timeout=timeout)
    rows = []
    for item in data.get("data", []):
        pdf = item.get("openAccessPdf") or {}
        rows.append({
            "doi": item.get("doi", ""),
            "title": item.get("title", ""),
            "abstract": item.get("abstract", ""),
            "year": item.get("year"),
            "venue": item.get("venue", ""),
            "url": item.get("url", ""),
            "full_text_url": pdf.get("url", "") if pdf else "",
            "oa_status": "open" if pdf and pdf.get("url") else "unknown",
            "license": "",
            "source_id": item.get("paperId", ""),
        })
    return rows


SEARCHERS = {
    "crossref": _search_crossref,
    "openalex": _search_openalex,
    "semantic_scholar": _search_semantic_scholar,
    "europe_pmc": _search_europe_pmc,
    "arxiv": _search_arxiv,
}


def _score_article(article, query_terms):
    title_norm = normalize_alias(article.get("title"))
    abstract_norm = normalize_alias(article.get("abstract"))
    terms = [term for term in query_terms if normalize_alias(term)]
    title_hits = [term for term in terms if normalize_alias(term) in title_norm]
    abstract_hits = [term for term in terms if normalize_alias(term) in abstract_norm]
    keyword_hits = [kw for kw in POLYMER_KEYWORDS if normalize_alias(kw) in title_norm or normalize_alias(kw) in abstract_norm]
    year = article.get("year") or 0
    try:
        year = int(year)
    except (TypeError, ValueError):
        year = 0
    recency = 0.08 if year >= 2020 else 0.03 if year >= 2010 else 0.0
    breakdown = {
        "title_alias_match": min(0.35, 0.12 * len(title_hits)),
        "abstract_alias_match": min(0.25, 0.06 * len(abstract_hits)),
        "polymer_keyword_score": min(0.20, 0.03 * len(keyword_hits)),
        "kg_keyword_score": min(0.10, 0.04 * len(set(title_hits + abstract_hits))),
        "oa_bonus": 0.12 if article.get("full_text_available") and article.get("oa_status") == "open" else 0.0,
        "recency_bonus": recency,
        "source_bonus": SOURCE_BONUS.get(article.get("source"), 0.0),
    }
    score = sum(breakdown.values())
    return round(score, 6), breakdown, title_hits, abstract_hits


def _dedupe_articles(articles):
    deduped = {}
    for article in articles:
        key = None
        if article.get("doi"):
            key = ("doi", article["doi"].lower())
        elif article.get("title"):
            key = ("title", normalize_alias(article["title"]))
        elif article.get("url"):
            key = ("url", article["url"])
        elif article.get("source_id"):
            key = ("source_id", article.get("source"), article["source_id"])
        else:
            key = ("article_id", article["article_id"])
        current = deduped.get(key)
        if current is None or article.get("article_score", 0) > current.get("article_score", 0):
            deduped[key] = article
        else:
            current["matched_repeat_unit_ids"] = sorted(set(current.get("matched_repeat_unit_ids", []) + article.get("matched_repeat_unit_ids", [])))
            current["matched_queries"] = sorted(set(current.get("matched_queries", []) + article.get("matched_queries", [])))
    return list(deduped.values())


def _write_metadata_only(article, documents_dir):
    path = Path(documents_dir) / _safe_filename("META_" + article["article_id"], ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join([
        f"Title: {article.get('title', '')}",
        f"DOI: {article.get('doi', '')}",
        f"Year: {article.get('year', '')}",
        f"Venue: {article.get('venue', '')}",
        f"Source: {article.get('source', '')}",
        f"OA status: {article.get('oa_status', '')}",
        f"Full text available: {str(article.get('full_text_available', False)).lower()}",
        "",
        "Abstract:",
        article.get("abstract", ""),
        "",
    ])
    path.write_text(content, encoding="utf-8")
    return str(path)


def _download(article, documents_dir, timeout=30):
    url = article.get("full_text_url")
    if not url:
        raise ValueError("missing full_text_url")
    suffix = ".pdf" if ".pdf" in url.lower() else ".xml" if "xml" in url.lower() else ".html"
    path = Path(documents_dir) / _safe_filename(article["article_id"], suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Uni-Poly-KG/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        content_type = response.headers.get("content-type", "").lower()
    path.write_bytes(data)
    if "pdf" in content_type or suffix == ".pdf":
        kind = "pdf"
    elif "xml" in content_type or suffix == ".xml":
        kind = "xml"
    else:
        kind = "html"
    return str(path), kind


def retrieve_literature(repeat_units_path, mapping_candidates_path, entity_aliases_path, output_dir, documents_dir,
                        sources="crossref,openalex,semantic_scholar,europe_pmc,arxiv", max_repeat_units=None,
                        max_queries_per_repeat_unit=5, max_results_per_query=10, max_articles_per_repeat_unit=5,
                        max_downloads_per_repeat_unit=3, max_total_downloads=100, min_article_score=0.4,
                        require_oa_for_download=True, metadata_only=False, timeout=20, sleep_seconds=0.2):
    del entity_aliases_path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(documents_dir).mkdir(parents=True, exist_ok=True)
    queries = build_retrieval_queries(repeat_units_path, mapping_candidates_path, max_repeat_units, max_queries_per_repeat_unit)
    write_jsonl(os.path.join(output_dir, "retrieval_queries.jsonl"), queries)
    enabled = [item.strip() for item in str(sources).split(",") if item.strip()]
    skipped = [item for item in enabled if item not in SEARCHERS]
    enabled = [item for item in enabled if item in SEARCHERS]
    _log(f"prepared queries={len(queries)} enabled_sources={enabled} skipped_sources={skipped}")
    _log(f"limits max_results_per_query={max_results_per_query} max_articles_per_repeat_unit={max_articles_per_repeat_unit} max_downloads_per_repeat_unit={max_downloads_per_repeat_unit} max_total_downloads={max_total_downloads} timeout={timeout}")
    raw_articles = []
    warnings = []
    total_requests = len(queries) * len(enabled)
    request_index = 0
    requests = [(query_index, query, source) for query_index, query in enumerate(queries, start=1) for source in enabled]
    for query_index, query, source in _progress(requests, total=len(requests), desc="retrieval metadata", unit="request"):
        query_text = str(query.get("query", ""))[:160]
        request_index += 1
        _log(f"metadata request {request_index}/{total_requests} query={query_index}/{len(queries)} source={source} repeat_unit_id={query.get('repeat_unit_id')} query_text={query_text!r} starting")
        started = time.time()
        try:
            rows = SEARCHERS[source](query, max_results_per_query, timeout)
            _log(f"metadata request {request_index}/{total_requests} source={source} ending rows={len(rows)} elapsed={time.time() - started:.2f}s")
            for item in rows:
                article = _normalize_article(source, item, query)
                score, breakdown, title_hits, abstract_hits = _score_article(article, query["query_terms"])
                article.update({
                    "matched_aliases": sorted(set(title_hits + abstract_hits)),
                    "matched_abbreviations": [term for term in query["query_terms"] if term.isupper() and term in title_hits + abstract_hits],
                    "matched_polymer_classes": [term for term in query["query_terms"] if normalize_alias("poly") in normalize_alias(term) or term.lower().startswith("poly")],
                    "article_score": score,
                    "score_breakdown": breakdown,
                    "filter_status": "raw",
                    "filter_reason": "metadata_search_result",
                })
                raw_articles.append(article)
        except Exception as exc:
            _log(f"metadata request {request_index}/{total_requests} source={source} failed elapsed={time.time() - started:.2f}s error={exc}")
            warnings.append(f"{source}_query_failed:{query['query_id']}:{exc}")
        time.sleep(sleep_seconds)
    _log(f"metadata search completed raw_articles={len(raw_articles)} warnings={len(warnings)}")
    deduped = _dedupe_articles(raw_articles)
    _log(f"dedupe completed deduplicated_articles={len(deduped)}")
    selected_by_ru = {}
    for article in sorted(deduped, key=lambda row: row.get("article_score", 0), reverse=True):
        if article["article_score"] < min_article_score:
            article["filter_status"] = "filtered_out"
            article["filter_reason"] = "below_min_article_score"
            continue
        kept = False
        for repeat_unit_id in article.get("matched_repeat_unit_ids", []):
            bucket = selected_by_ru.setdefault(repeat_unit_id, [])
            if len(bucket) < max_articles_per_repeat_unit:
                bucket.append(article["article_id"])
                kept = True
        article["filter_status"] = "selected" if kept else "filtered_out"
        article["filter_reason"] = "selected_by_score" if kept else "per_repeat_unit_article_limit"
    selected = [article for article in deduped if article.get("filter_status") == "selected"]
    _log(f"article filtering completed selected={len(selected)} filtered_out={sum(1 for article in deduped if article.get('filter_status') == 'filtered_out')}")
    downloads_by_ru = {}
    total_downloads = 0
    manifest = []
    sorted_selected = sorted(selected, key=lambda row: row.get("article_score", 0), reverse=True)
    for article_index, article in _progress(enumerate(sorted_selected, start=1), total=len(sorted_selected), desc="retrieval download", unit="article"):
        can_download = (
            not metadata_only
            and article.get("full_text_available")
            and (not require_oa_for_download or article.get("oa_status") == "open")
            and total_downloads < max_total_downloads
        )
        ru_allowed = False
        for repeat_unit_id in article.get("matched_repeat_unit_ids", []):
            if downloads_by_ru.get(repeat_unit_id, 0) < max_downloads_per_repeat_unit:
                ru_allowed = True
                break
        if can_download and ru_allowed:
            _log(f"download {article_index}/{len(sorted_selected)} article_id={article.get('article_id')} source={article.get('source')} score={article.get('article_score')} url={article.get('full_text_url')} starting")
            started = time.time()
            try:
                local_path, content_type = _download(article, documents_dir, timeout=timeout)
                status, error = "downloaded", ""
                _log(f"download {article_index}/{len(sorted_selected)} article_id={article.get('article_id')} ending content_type={content_type} path={local_path} elapsed={time.time() - started:.2f}s")
                total_downloads += 1
                for repeat_unit_id in article.get("matched_repeat_unit_ids", []):
                    downloads_by_ru[repeat_unit_id] = downloads_by_ru.get(repeat_unit_id, 0) + 1
                article["filter_status"] = "download_selected"
                article["local_path"] = local_path
            except Exception as exc:
                local_path = _write_metadata_only(article, documents_dir)
                content_type, status, error = "metadata_only", "failed", str(exc)
                article["filter_status"] = "metadata_only"
                warnings.append(f"download_failed:{article['article_id']}:{exc}")
                _log(f"download {article_index}/{len(sorted_selected)} article_id={article.get('article_id')} failed error={exc}; wrote metadata_only path={local_path}")
        else:
            local_path = _write_metadata_only(article, documents_dir)
            content_type = "metadata_only"
            status = "metadata_only"
            error = ""
            reason = []
            if metadata_only:
                reason.append("metadata_only_enabled")
            if not article.get("full_text_available"):
                reason.append("no_full_text_url")
            if require_oa_for_download and article.get("oa_status") != "open":
                reason.append(f"oa_status={article.get('oa_status')}")
            if total_downloads >= max_total_downloads:
                reason.append("max_total_downloads_reached")
            if not ru_allowed:
                reason.append("max_downloads_per_repeat_unit_reached")
            _log(f"download {article_index}/{len(sorted_selected)} article_id={article.get('article_id')} skipped reason={','.join(reason) or 'not_downloadable'} metadata_path={local_path}")
            article["filter_status"] = "metadata_only"
            article["local_path"] = local_path
        manifest.append({
            "article_id": article["article_id"],
            "doi": article.get("doi", ""),
            "source": article.get("source", ""),
            "download_url": article.get("full_text_url", ""),
            "local_path": local_path,
            "content_type": content_type,
            "status": status,
            "error": error,
            "license": article.get("license", ""),
            "full_text_available": article.get("full_text_available", False),
        })
    _log(f"download phase completed downloaded={sum(1 for item in manifest if item['status'] == 'downloaded')} metadata_only={sum(1 for item in manifest if item['content_type'] == 'metadata_only')} failed={sum(1 for item in manifest if item['status'] == 'failed')}")
    write_jsonl(os.path.join(output_dir, "retrieved_articles.jsonl"), sorted(deduped, key=lambda row: (-row.get("article_score", 0), row.get("article_id", ""))))
    write_jsonl(os.path.join(output_dir, "download_manifest.jsonl"), manifest)
    _log(f"wrote retrieved_articles={os.path.join(output_dir, 'retrieved_articles.jsonl')}")
    _log(f"wrote download_manifest={os.path.join(output_dir, 'download_manifest.jsonl')}")
    report = {
        "retrieval_mode": "online",
        "query_count": len(queries),
        "raw_search_result_count": len(raw_articles),
        "deduplicated_article_count": len(deduped),
        "filtered_out_article_count": sum(1 for article in deduped if article.get("filter_reason") != "selected_by_score"),
        "selected_article_count": len(selected),
        "download_selected_count": sum(1 for item in manifest if item["status"] == "downloaded"),
        "downloaded_full_text_count": sum(1 for item in manifest if item["status"] == "downloaded"),
        "metadata_only_count": sum(1 for item in manifest if item["content_type"] == "metadata_only"),
        "failed_download_count": sum(1 for item in manifest if item["status"] == "failed"),
        "sources_enabled": enabled,
        "sources_skipped": skipped,
        "max_total_downloads": max_total_downloads,
        "min_article_score": min_article_score,
        "warnings": warnings,
    }
    write_json(os.path.join(output_dir, "retrieval_report.json"), report)
    _log(f"wrote retrieval_report={os.path.join(output_dir, 'retrieval_report.json')}")
    return report


import html
import os
import re
from pathlib import Path

from .io_utils import normalize_text, stable_hash, stable_id, write_jsonl


SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".xml", ".pdf"}


def _read_document(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            pages = []
            for page_number, page in enumerate(PdfReader(str(path)).pages, 1):
                pages.append((page_number, page.extract_text() or ""))
            return "pdf", pages
        except ImportError as exc:
            raise RuntimeError("PDF parsing requires pypdf; install it or provide HTML/XML/TXT") from exc
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm", ".xml"}:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "xml" if suffix == ".xml" else "html.parser")
            for node in soup(["script", "style"]):
                node.decompose()
            blocks = []
            for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "caption", "tr"]):
                value = normalize_text(node.get_text(" ", strip=True))
                if value:
                    blocks.append(value)
            text = "\n\n".join(blocks) or soup.get_text("\n")
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html.unescape(text))
    return suffix.lstrip("."), [(None, text)]


def _token_chunks(text, target_tokens=450, overlap_tokens=75):
    paragraphs = [normalize_text(part) for part in re.split(r"\n\s*\n+", text) if normalize_text(part)]
    words = []
    paragraph_boundaries = []
    for paragraph_id, paragraph in enumerate(paragraphs):
        start = len(words)
        words.extend(paragraph.split())
        paragraph_boundaries.append((start, len(words), paragraph_id))
    if not words:
        return []
    step = max(1, target_tokens - overlap_tokens)
    chunks = []
    for start in range(0, len(words), step):
        end = min(len(words), start + target_tokens)
        pids = [pid for pstart, pend, pid in paragraph_boundaries if pend > start and pstart < end]
        chunks.append((start, end, pids, " ".join(words[start:end])))
        if end == len(words):
            break
    return chunks


def parse_documents(input_dir, output_path, target_tokens=450, overlap_tokens=75):
    rows = []
    for path in sorted(Path(input_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        article_id = stable_id("art", path.name, path.stat().st_size, stable_hash(path.read_bytes(), length=24))
        source_type, pages = _read_document(path)
        order = 0
        for page, page_text in pages:
            section = None
            for start, end, paragraph_ids, text in _token_chunks(page_text, target_tokens, overlap_tokens):
                chunk_id = stable_id("chunk", article_id, order, text, length=20)
                rows.append({
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "article_title": path.stem,
                    "source_path": str(path),
                    "source_type": source_type,
                    "section": section,
                    "page": page,
                    "paragraph_ids": paragraph_ids,
                    "order_start": start,
                    "order_end": end,
                    "text": text,
                    "text_hash": stable_hash(text, length=24),
                    "document_order": order,
                })
                order += 1
    write_jsonl(output_path, rows)
    return len(rows)


import math
import os
import re
from collections import Counter, defaultdict

from .io_utils import normalize_alias, read_jsonl, stable_id, write_jsonl


FACT_KEYWORDS = {
    "identity": ["polymer", "sample", "material", "abbreviation", "designated", "denoted"],
    "composition": ["composition", "copolymer", "blend", "composite", "monomer", "ratio", "mol%", "wt%", "feed"],
    "sequence": ["random", "statistical", "alternating", "block", "gradient", "graft", "sequence"],
    "architecture": ["linear", "branched", "star", "comb", "brush", "network", "crosslinked", "architecture"],
    "molecular_weight": ["molecular weight", "mn", "mw", "pdi", "dispersity", "gpc", "sec", "degree of polymerization"],
    "polymerization": ["polymerization", "synthesized", "prepared", "temperature", "solvent", "atmosphere", "anneal", "casting"],
}
SECTION_PRIORS = {
    "identity": ["abstract", "materials"], "composition": ["materials", "experimental"],
    "sequence": ["characterization", "results"], "architecture": ["results", "discussion"],
    "molecular_weight": ["characterization", "results", "table"],
    "polymerization": ["experimental", "methods", "synthesis"],
}


def _tokens(text):
    return re.findall(r"[a-z0-9%]+", str(text).lower())



def _load_retrieval_metadata(retrieval_dir=None):
    if not retrieval_dir:
        return {}
    import os
    manifest_path = os.path.join(retrieval_dir, "download_manifest.jsonl")
    articles_path = os.path.join(retrieval_dir, "retrieved_articles.jsonl")
    articles = {row.get("article_id"): row for row in read_jsonl(articles_path)}
    metadata = {}
    for row in read_jsonl(manifest_path):
        article = articles.get(row.get("article_id"), {})
        local_path = row.get("local_path")
        if local_path:
            metadata[os.path.abspath(local_path)] = {
                "retrieval_article_score": article.get("article_score", 0.0),
                "retrieval_source": article.get("source", row.get("source", "")),
                "metadata_only": row.get("content_type") == "metadata_only",
                "full_text_available": bool(row.get("full_text_available")),
            }
    return metadata

def _load_mapping_terms(mapping_path=None, aliases_path=None):
    if not mapping_path:
        return []
    rows = read_jsonl(mapping_path)
    terms = []
    for row in rows:
        confidence = float(row.get("confidence") or 0)
        source_type = row.get("source_type", "mapping")
        base = {
            "repeat_unit_id": row.get("repeat_unit_id"),
            "mapping_source_type": source_type,
            "mapping_confidence": confidence,
        }
        for field, term_type in (("raw_smiles", "raw_smiles"), ("canonical_smiles", "canonical_smiles")):
            value = row.get(field)
            if value and value != "unknown":
                terms.append({**base, "term": str(value), "term_type": term_type})
        for value in row.get("aliases") or []:
            terms.append({**base, "term": str(value), "term_type": "alias"})
        for candidate in row.get("polymer_class_candidates", []):
            for value in [candidate.get("canonical_name"), candidate.get("polymer_class"), candidate.get("polymer_family")]:
                if value and value != "unknown":
                    terms.append({**base, "term": str(value), "term_type": "polymer_class"})
            for value in candidate.get("aliases") or []:
                terms.append({**base, "term": str(value), "term_type": "alias"})
    seen = set()
    unique = []
    for item in terms:
        key = (normalize_alias(item["term"]), item["term_type"], item.get("repeat_unit_id"))
        if key[0] and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _match_mapping_terms(text, mapping_terms):
    lower = str(text).lower()
    normalized = normalize_alias(text)
    matches = []
    for item in mapping_terms:
        term = item["term"]
        norm = normalize_alias(term)
        if not norm:
            continue
        matched = False
        if item["term_type"] in {"raw_smiles", "canonical_smiles"}:
            matched = term in text
        elif len(norm) <= 4:
            matched = re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None
        else:
            matched = norm in normalized or term.lower() in lower
        if matched:
            matches.append(item)
    return matches


def recall_chunks(source_chunks_path, output_path, bm25_top_k=10, dense_top_k=10,
                  merged_top_k=8, neighbor_window=1, max_iterations=2,
                  mapping_path=None, aliases_path=None, sample_conditioned_top_k=5, retrieval_dir=None):
    del dense_top_k, max_iterations, aliases_path
    chunks = read_jsonl(source_chunks_path)
    mapping_terms = _load_mapping_terms(mapping_path)
    retrieval_metadata = _load_retrieval_metadata(retrieval_dir)
    by_article = defaultdict(list)
    for chunk in chunks:
        by_article[chunk["article_id"]].append(chunk)
    for article_chunks in by_article.values():
        article_chunks.sort(key=lambda item: item.get("document_order", 0))

    output_by_id = {}
    for article_id, article_chunks in by_article.items():
        documents = [_tokens(chunk["text"]) for chunk in article_chunks]
        df = Counter(token for doc in documents for token in set(doc))
        avg_len = sum(map(len, documents)) / max(1, len(documents))
        for fact_type, keywords in FACT_KEYWORDS.items():
            query_tokens = _tokens(" ".join(keywords))
            scored = []
            for index, (chunk, doc) in enumerate(zip(article_chunks, documents)):
                counts = Counter(doc)
                bm25 = 0.0
                for token in query_tokens:
                    freq = counts[token]
                    if not freq:
                        continue
                    idf = math.log(1 + (len(documents) - df[token] + 0.5) / (df[token] + 0.5))
                    bm25 += idf * freq * 2.2 / (freq + 1.2 * (1 - 0.75 + 0.75 * len(doc) / max(avg_len, 1)))
                lower = chunk["text"].lower()
                matches = [keyword for keyword in keywords if keyword in lower]
                mapping_matches = _match_mapping_terms(chunk["text"], mapping_terms)
                keyword_score = len(matches) / max(1, len(keywords))
                section = str(chunk.get("section") or "").lower()
                section_score = float(any(term in section for term in SECTION_PRIORS[fact_type]))
                table_score = float(any(term in lower for term in ("table", "caption", "figure")))
                mapping_score = min(1.0, len(mapping_matches) / 3.0)
                score = 0.30 * min(1.0, bm25 / 5.0) + 0.20 * keyword_score + 0.10 * section_score + 0.05 * table_score + 0.25 * mapping_score
                if score > 0:
                    scored.append((score, index, bm25, keyword_score, section_score, table_score, matches, mapping_matches))
            top_n = min(bm25_top_k, merged_top_k + (sample_conditioned_top_k if mapping_terms else 0))
            for score, index, bm25, keyword_score, section_score, table_score, matches, mapping_matches in sorted(scored, reverse=True)[:top_n]:
                context = []
                for neighbor in range(max(0, index - neighbor_window), min(len(article_chunks), index + neighbor_window + 1)):
                    context.append(article_chunks[neighbor]["chunk_id"])
                chunk = article_chunks[index]
                retrieval_info = retrieval_metadata.get(os.path.abspath(chunk.get("source_path", "")), {})
                if retrieval_info.get("retrieval_article_score"):
                    score += 0.05 * float(retrieval_info.get("retrieval_article_score") or 0)
                candidate_id = stable_id("cand", article_id, chunk["chunk_id"], fact_type)
                matched_aliases = [m["term"] for m in mapping_matches if m["term_type"] in {"alias", "common_name"}]
                matched_classes = [m["term"] for m in mapping_matches if m["term_type"] in {"polymer_class", "polymer_family"}]
                matched_abbreviations = [m["term"] for m in mapping_matches if m["term_type"] == "abbreviation"]
                mapping_confidence = max([float(m.get("mapping_confidence") or 0) for m in mapping_matches] or [0.0])
                output_by_id[candidate_id] = {
                    "candidate_id": candidate_id,
                    "article_id": article_id,
                    "chunk_id": chunk["chunk_id"],
                    "fact_type": fact_type,
                    "score": round(score, 6),
                    "score_parts": {
                        "bm25": bm25,
                        "dense_similarity": None,
                        "keyword_coverage": keyword_score,
                        "section_prior": section_score,
                        "table_expansion": table_score,
                        "mapping_term_score": min(1.0, len(mapping_matches) / 3.0),
                        "matched_keywords": matches,
                    },
                    "matched_terms": sorted(set(matches + [m["term"] for m in mapping_matches])),
                    "matched_aliases": sorted(set(matched_aliases)),
                    "matched_polymer_classes": sorted(set(matched_classes)),
                    "matched_abbreviations": sorted(set(matched_abbreviations)),
                    "mapping_source_type": sorted(set(m.get("mapping_source_type", "") for m in mapping_matches if m.get("mapping_source_type"))),
                    "mapping_confidence": mapping_confidence,
                    "recall_reason": "sample_conditioned_mapping" if mapping_matches else "field_keyword_bm25",
                    "retrieval_article_score": retrieval_info.get("retrieval_article_score", 0.0),
                    "retrieval_source": retrieval_info.get("retrieval_source", ""),
                    "metadata_only": retrieval_info.get("metadata_only", False),
                    "full_text_available": retrieval_info.get("full_text_available", False),
                    "context_chunk_ids": context,
                    "neighboring_chunk_ids": context,
                    "iteration": 1,
                    "sample_candidate": None,
                    "warnings": ["dense_retrieval_unavailable"],
                }
    output = sorted(output_by_id.values(), key=lambda row: (-row["score"], row["article_id"], row["chunk_id"], row["fact_type"]))
    write_jsonl(output_path, output)
    return len(output)
