import html
import os
import re
from pathlib import Path

from .common import normalize_text, stable_hash, stable_id, write_jsonl


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

