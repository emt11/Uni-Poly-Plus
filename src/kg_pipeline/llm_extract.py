"""LLM providers, RU alias mapping, and literature extraction."""

import json
import os
import time
import urllib.error
import urllib.request

from .io_utils import assert_no_val_fields, read_csv, read_jsonl, stable_id, write_jsonl
from .article_validation import empty_article_document


def _progress(iterable, total=None, desc="progress"):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="item")
    except Exception:
        return iterable


PROVIDERS = {
    "qwen": {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key_env": "DASHSCOPE_API_KEY",
        "model_env": "QWEN_MODEL",
        "default_model": "qwen3.7-plus",
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

MAPPING_SYSTEM_PROMPT = """Role:
You are a polymer nomenclature engineer specializing in mapping polymer repeat-unit SMILES to
literature-search aliases.

Task:
Given only one raw repeat-unit SMILES, output exactly one likely English polymer name and,
only when available, common aliases for that polymer. Abbreviations can be used as aliases. Do
not use any information other than the supplied raw_smiles. Do not output broad polymer
classes or families such as polyester, polyamide, polymer, condensation polymer, or
fluoropolymer unless the broad term is the only defensible English polymer name. If the
raw_smiles is too ambiguous, set polymer_name to an empty string, aliases to an empty list,
confidence to 0.0. Return JSON only. polymer_name must contain exactly one English polymer
name when identifiable; otherwise use an empty string. aliases is optional and can contain
common English names or abbreviations. confidence must be between 0 and 1.

Example input:
{"raw_smiles": "*CC(*)C"}

Example output:
{
  "polymer_name": "polypropylene",
  "aliases": ["polypropene", "PP"],
  "confidence": 0.85
}"""

MOCK_ALIAS_RULES = [
    ("C(=O)N", ["polyamide"], 0.55),
    ("NC(=O)", ["polyamide"], 0.55),
    ("C(=O)O", ["polyester"], 0.40),
    ("OC(=O)", ["polyester"], 0.40),
    ("F", ["fluorinated polymer"], 0.35),
    ("Si", ["silicone"], 0.45),
    ("C=C", ["vinyl polymer"], 0.30),
]


def _extract_json(text):
    value = str(text or "").strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(value[start:end + 1])


def provider_model(provider, model=None):
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if model:
        return model
    config = PROVIDERS[provider]
    return os.environ.get(config["model_env"], config["default_model"])


def provider_key_env(provider):
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDERS[provider]["key_env"]


def call_provider(provider, messages, timeout=120, max_retries=2, sleep_seconds=1.0, model=None):
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["key_env"])
    if not api_key:
        raise RuntimeError(f"Missing API key: set {config['key_env']} in the environment")
    payload = {
        "model": provider_model(provider, model),
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    last_error = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            config["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"], body.get("usage", {})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = RuntimeError(f"{provider} HTTP {exc.code}: {detail}")
            if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = RuntimeError(f"{provider} request failed: {exc}")
        if attempt < max_retries:
            time.sleep(sleep_seconds * (2 ** attempt))
    raise last_error


def _mock_article_document(candidate, context):
    article_id = candidate["article_id"]
    document = empty_article_document(article_id)
    if context:
        chunk = context[0]
        sentence = str(chunk.get("text", "")).split(".")[0].strip()
        if sentence:
            evidence_id = stable_id("ev", candidate["candidate_id"], chunk["chunk_id"], length=20)
            document["evidence_records"].append({
                "evidence_id": evidence_id,
                "chunk_id": chunk["chunk_id"],
                "sentence": sentence,
            })
    document["warnings"] = ["mock_llm_extraction"]
    return document


def extract_articles(candidate_chunks_path, output_dir, provider, source_chunks_path=None, max_candidates=None,
                     mock_response=False, dry_run=False, timeout=120, max_retries=2, sleep_seconds=1.0, model=None):
    candidates = read_jsonl(candidate_chunks_path)
    source_chunks_path = source_chunks_path or os.path.join(os.path.dirname(candidate_chunks_path), "source_chunks.jsonl")
    chunks = {row["chunk_id"]: row for row in read_jsonl(source_chunks_path)}
    os.makedirs(output_dir, exist_ok=True)
    log_rows = []
    successes = 0
    raw_rows = []
    selected_candidates = candidates[:max_candidates]
    for candidate in _progress(selected_candidates, total=len(selected_candidates), desc="extraction"):
        context = [chunks[chunk_id] for chunk_id in candidate.get("context_chunk_ids", []) if chunk_id in chunks]
        if not context and candidate["chunk_id"] in chunks:
            context = [chunks[candidate["chunk_id"]]]
        article_id = candidate["article_id"]
        payload = {
            "article_id": article_id,
            "fact_type": candidate["fact_type"],
            "chunks": [
                {"chunk_id": item["chunk_id"], "section": item.get("section"), "page": item.get("page"), "text": item["text"]}
                for item in context
            ],
        }
        assert_no_val_fields(payload, "extraction_prompt")
        started = time.time()
        usage = {}
        try:
            if dry_run:
                status, error = "dry_run", None
                log_rows.append({
                    "call_id": stable_id("call", candidate["candidate_id"], provider),
                    "candidate_id": candidate["candidate_id"], "provider": provider,
                    "model": provider_model(provider, model), "status": status, "error": error, "usage": usage,
                    "elapsed_time": round(time.time() - started, 3),
                })
                continue
            if mock_response:
                document = _mock_article_document(candidate, context)
            else:
                content, usage = call_provider(provider, [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ], timeout=timeout, max_retries=max_retries, sleep_seconds=sleep_seconds, model=model)
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
            status, error = "failed", str(exc)
        log_rows.append({
            "call_id": stable_id("call", candidate["candidate_id"], provider),
            "candidate_id": candidate["candidate_id"], "provider": provider,
            "model": provider_model(provider, model), "status": status, "error": error, "usage": usage,
            "elapsed_time": round(time.time() - started, 3),
        })
    write_jsonl(os.path.join(output_dir, "extractions.jsonl"), raw_rows)
    write_jsonl(os.path.join(output_dir, "calls.jsonl"), log_rows)
    return successes


def _mapping_payload(unit):
    payload = {
        "raw_smiles": unit.get("raw_smiles", ""),
    }
    assert_no_val_fields(payload, "mapping_prompt")
    return payload


def _mock_mapping(unit, provider):
    del provider
    text = unit.get("raw_smiles", "")
    aliases, confidence = [], 0.20
    for token, rule_aliases, rule_confidence in MOCK_ALIAS_RULES:
        if token.lower() in text.lower():
            aliases, confidence = rule_aliases, rule_confidence
            break
    return {
        "repeat_unit_id": unit["repeat_unit_id"],
        "raw_smiles": unit.get("raw_smiles", ""),
        "canonical_smiles": unit.get("canonical_smiles", ""),
        "aliases": aliases,
        "confidence": confidence,
    }


def _normalize_mapping_response(unit, response, provider, model):
    del provider, model
    polymer_name = str(response.get("polymer_name") or "").strip()
    aliases = response.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    aliases = sorted(set(str(item).strip() for item in [polymer_name] + list(aliases) if str(item).strip()))
    try:
        confidence = max(0.0, min(1.0, float(response.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "repeat_unit_id": unit["repeat_unit_id"],
        "raw_smiles": unit.get("raw_smiles", ""),
        "canonical_smiles": unit.get("canonical_smiles", ""),
        "aliases": aliases,
        "confidence": confidence,
    }


def generate_llm_polymer_mapping(repeat_units_path, output_dir, provider, max_units=None,
                                 mock_response=False, dry_run=False, overwrite=False,
                                 sleep_seconds=1.0, timeout=120, max_retries=2, model=None):
    mapping_path = os.path.join(output_dir, "polymer_class_candidates.jsonl")
    if os.path.exists(mapping_path) and not overwrite:
        raise FileExistsError(f"{mapping_path} exists; pass --overwrite to replace it")
    units = read_csv(repeat_units_path)[:max_units]
    model = "mock" if mock_response else provider_model(provider, model)
    mapped = []
    calls = []
    os.makedirs(os.path.join(output_dir, "mapping"), exist_ok=True)
    for unit in _progress(units, total=len(units), desc="mapping"):
        started = time.time()
        status, error, usage = "ok", None, {}
        try:
            if dry_run:
                status = "dry_run"
                mapped_row = _mock_mapping(unit, provider)
            elif mock_response:
                mapped_row = _mock_mapping(unit, provider)
            else:
                payload = _mapping_payload(unit)
                content, usage = call_provider(provider, [
                    {"role": "system", "content": MAPPING_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ], timeout=timeout, max_retries=max_retries, sleep_seconds=sleep_seconds, model=model)
                mapped_row = _normalize_mapping_response(unit, _extract_json(content), provider, model)
            mapped.append(mapped_row)
        except Exception as exc:
            status, error = "failed", str(exc)
            calls.append({
                "call_id": stable_id("mapcall", unit.get("repeat_unit_id"), provider),
                "repeat_unit_id": unit.get("repeat_unit_id"), "provider": provider, "model": model,
                "status": status, "error": error, "usage": usage,
                "elapsed_time": round(time.time() - started, 3),
            })
            write_jsonl(os.path.join(output_dir, "mapping", "calls.jsonl"), calls)
            raise
        calls.append({
            "call_id": stable_id("mapcall", unit.get("repeat_unit_id"), provider),
            "repeat_unit_id": unit.get("repeat_unit_id"), "provider": provider, "model": model,
            "status": status, "error": error, "usage": usage,
            "elapsed_time": round(time.time() - started, 3),
        })
        if sleep_seconds and not mock_response and not dry_run:
            time.sleep(sleep_seconds)
    count = write_mapping_outputs(repeat_units_path, output_dir, mapped, f"llm_{provider}")
    write_jsonl(os.path.join(output_dir, "mapping", "calls.jsonl"), calls)
    return count


import os

from .io_utils import read_csv, write_jsonl


def _aliases_from_row(row):
    values = row.get("aliases") or []
    return sorted(set(str(item).strip() for item in values if str(item).strip()))


def write_mapping_outputs(repeat_units_path, output_dir, mapped_rows, mapping_method=None):
    del mapping_method
    units = {row["repeat_unit_id"]: row for row in read_csv(repeat_units_path)}
    output_rows = []
    for mapped in mapped_rows:
        unit = units[mapped["repeat_unit_id"]]
        confidence = max(0.0, min(1.0, float(mapped.get("confidence", 0.0))))
        output_rows.append({
            "repeat_unit_id": unit["repeat_unit_id"],
            "raw_smiles": unit.get("raw_smiles", ""),
            "canonical_smiles": unit.get("canonical_smiles", ""),
            "aliases": _aliases_from_row(mapped),
            "confidence": confidence,
        })
    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(f"{output_dir}/polymer_class_candidates.jsonl", output_rows)
    return len(output_rows)
