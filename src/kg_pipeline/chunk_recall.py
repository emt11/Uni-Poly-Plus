import math
import re
from collections import Counter, defaultdict

from .common import read_jsonl, stable_id, write_jsonl


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


def recall_chunks(source_chunks_path, output_path, bm25_top_k=10, dense_top_k=10,
                  merged_top_k=8, neighbor_window=1, max_iterations=2):
    del dense_top_k, max_iterations
    chunks = read_jsonl(source_chunks_path)
    by_article = defaultdict(list)
    for chunk in chunks:
        by_article[chunk["article_id"]].append(chunk)
    for article_chunks in by_article.values():
        article_chunks.sort(key=lambda item: item.get("document_order", 0))

    output = []
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
                keyword_score = len(matches) / max(1, len(keywords))
                section = str(chunk.get("section") or "").lower()
                section_score = float(any(term in section for term in SECTION_PRIORS[fact_type]))
                table_score = float(any(term in lower for term in ("table", "caption", "figure")))
                score = 0.30 * min(1.0, bm25 / 5.0) + 0.20 * keyword_score + 0.10 * section_score + 0.05 * table_score
                if score > 0:
                    scored.append((score, index, bm25, keyword_score, section_score, table_score, matches))
            for score, index, bm25, keyword_score, section_score, table_score, matches in sorted(scored, reverse=True)[:min(bm25_top_k, merged_top_k)]:
                context = []
                for neighbor in range(max(0, index - neighbor_window), min(len(article_chunks), index + neighbor_window + 1)):
                    context.append(article_chunks[neighbor]["chunk_id"])
                chunk = article_chunks[index]
                output.append({
                    "candidate_id": stable_id("cand", article_id, chunk["chunk_id"], fact_type),
                    "article_id": article_id,
                    "chunk_id": chunk["chunk_id"],
                    "fact_type": fact_type,
                    "score": round(score, 6),
                    "score_parts": {"bm25": bm25, "dense_similarity": None, "keyword_coverage": keyword_score, "section_prior": section_score, "table_expansion": table_score, "matched_keywords": matches},
                    "context_chunk_ids": context,
                    "iteration": 1,
                    "sample_candidate": None,
                    "warnings": ["dense_retrieval_unavailable"] if True else [],
                })
    write_jsonl(output_path, output)
    return len(output)
