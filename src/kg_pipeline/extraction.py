import json
import os
import time
import urllib.error
import urllib.request

from .common import assert_no_val_fields, read_jsonl, stable_id, write_jsonl
from .schemas import empty_article_document


PROVIDERS = {
    "qwen": {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key_env": "DASHSCOPE_API_KEY",
        "model_env": "QWEN_MODEL",
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
}


SYSTEM_PROMPT = """You extract polymer literature facts. Return JSON only.
Use schema_version 2.0 with article, evidence_records, literature_samples,
dataset_links, and warnings. Extract only facts explicitly stated in the supplied
chunks. Every assertion, measurement, event, and identity must cite an evidence
record whose sentence is copied verbatim from the chunks. Use [] when no fact is
present. Never infer sequence distribution or chain architecture. Multiple monomers
in PA66/PET-like condensation chemistry do not imply random/block/graft copolymer.
Do not create all-null assertions, measurements, or events. dataset_links must be []."""


def _extract_json(text):
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(value[start:end + 1])


def call_provider(provider, messages, timeout=120):
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["key_env"])
    if not api_key:
        raise RuntimeError(f"Missing environment variable {config['key_env']}")
    payload = {
        "model": os.environ.get(config["model_env"], config["default_model"]),
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        config["url"], data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"{provider} HTTP {exc.code}: {detail}") from exc
    return body["choices"][0]["message"]["content"], body.get("usage", {})


def extract_articles(candidate_chunks_path, output_dir, provider, source_chunks_path=None, max_candidates=None):
    candidates = read_jsonl(candidate_chunks_path)
    source_chunks_path = source_chunks_path or os.path.join(os.path.dirname(candidate_chunks_path), "source_chunks.jsonl")
    chunks = {row["chunk_id"]: row for row in read_jsonl(source_chunks_path)}
    os.makedirs(output_dir, exist_ok=True)
    log_rows = []
    successes = 0
    raw_rows = []
    for candidate in candidates[:max_candidates]:
        context = [chunks[chunk_id] for chunk_id in candidate.get("context_chunk_ids", []) if chunk_id in chunks]
        if not context and candidate["chunk_id"] in chunks:
            context = [chunks[candidate["chunk_id"]]]
        article_id = candidate["article_id"]
        payload = {
            "article_id": article_id,
            "fact_type": candidate["fact_type"],
            "chunks": [{"chunk_id": item["chunk_id"], "section": item.get("section"), "page": item.get("page"), "text": item["text"]} for item in context],
        }
        assert_no_val_fields(payload, "extraction_prompt")
        started = time.time()
        try:
            content, usage = call_provider(provider, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ])
            document = _extract_json(content)
            document.setdefault("schema_version", "2.0")
            document["article"] = document.get("article") or empty_article_document(article_id)["article"]
            document["article"]["article_id"] = article_id
            for field in ("evidence_records", "literature_samples", "dataset_links", "warnings"):
                document.setdefault(field, [])
            output_path = os.path.join(output_dir, f"{candidate['candidate_id']}.json")
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
            raw_rows.append({"candidate_id": candidate["candidate_id"], "document": document})
            successes += 1
            status, error = "ok", None
        except Exception as exc:
            usage = {}
            status, error = "failed", str(exc)
        log_rows.append({
            "call_id": stable_id("call", candidate["candidate_id"], provider),
            "candidate_id": candidate["candidate_id"], "provider": provider,
            "status": status, "error": error, "usage": usage,
            "elapsed_seconds": round(time.time() - started, 3),
        })
    write_jsonl(os.path.join(output_dir, "extractions.jsonl"), raw_rows)
    write_jsonl(os.path.join(output_dir, "calls.jsonl"), log_rows)
    return successes


def generate_llm_polymer_mapping(repeat_units_path, output_dir, provider, max_units=None):
    from .common import read_csv
    from .polymer_classes import write_mapping_outputs

    mapped = []
    for unit in read_csv(repeat_units_path)[:max_units]:
        prompt = {
            "repeat_unit_id": unit["repeat_unit_id"],
            "raw_smiles": unit["raw_smiles"],
            "canonical_smiles": unit["canonical_smiles"],
            "instruction": "Return one conservative temporary PolymerClass candidate or unknown. Do not infer random/block/graft from fixed condensation repeat units.",
            "output_schema": {"polymer_class_id": "pc_slug", "canonical_name": "name", "polymer_family": "family", "aliases": [], "composition_type": "enum", "confidence": 0.0, "warnings": []},
        }
        assert_no_val_fields(prompt, "mapping_prompt")
        content, _ = call_provider(provider, [
            {"role": "system", "content": "Return JSON only. This is an unverified temporary mapping; use unknown when uncertain."},
            {"role": "user", "content": json.dumps(prompt)},
        ])
        candidate = _extract_json(content)
        mapped.append({
            "repeat_unit_id": unit["repeat_unit_id"],
            "polymer_class_candidates": [{key: candidate.get(key) for key in ("polymer_class_id", "canonical_name", "polymer_family", "aliases", "composition_type")}],
            "confidence": min(float(candidate.get("confidence", 0.4)), 0.7),
            "warnings": candidate.get("warnings", []),
        })
    return write_mapping_outputs(repeat_units_path, output_dir, mapped, f"llm_{provider}", "llm_temporary_mapping")

