# =========================
# Imports
# =========================
import argparse
import csv
import json
import math
import re
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from langchain_ollama import OllamaEmbeddings

# Optional OpenAI support (only used if a model starts with "openai:")
try:
    from langchain_openai import OpenAIEmbeddings

    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# =========================
# Config (edit here)
# =========================
DOC_LIMIT_DEFAULT = 5000  # default number of docs to keep in corpus
K = 10  # @K for metrics
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_BATCH_SIZE = 64
OPENAI_EMBED_BATCH_SIZE = 128
FAILED_DOCS_MD_PATH = Path("failed_doc.md")
FAILED_DOC_BEHAVIOR = "skip-doc"  # one of: skip-doc, zero-vector, fail
# MODELS = [
#     "embeddinggemma",
#     "qwen3-embedding:0.6b",
#     "bge-large",
#     "nomic-embed-text",
#     "mxbai-embed-large"
#     # "openai:text-embedding-3-small",  # requires OPENAI_API_KEY + extra deps
# ]
MODELS = ["bge-m3", "qwen3-embedding:0.6b", "qwen3-embedding", "embeddinggemma"]
USE_TITLE_PLUS_TEXT = False  # set True to concatenate "title: text" as the doc string
RESULTS_MD_PATH = Path("result.md")
DOC_EMBED_CACHE_DIR = Path(".cache/doc_embeddings")


# =========================
# Metrics
# =========================
def recall_at_k(ranks: List[int], k: int) -> float:
    hits = sum(1 for r in ranks if r <= k)
    return hits / len(ranks) if ranks else 0.0


def mrr_at_k(ranks: List[int], k: int) -> float:
    rr = [1.0 / r if r <= k else 0.0 for r in ranks]
    return float(np.mean(rr)) if rr else 0.0


def ndcg_at_k(gains_lists: List[List[int]], k: int) -> float:
    def dcg(gains):
        return sum((g / math.log2(i + 2)) for i, g in enumerate(gains[:k]))

    vals = []
    for gains in gains_lists:
        ideal = sorted(gains, reverse=True)
        idcg = dcg(ideal)
        vals.append(dcg(gains) / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


# =========================
# Dataset loaders (Thai)
# =========================
def load_full_miracl_th_corpus(
    json_paths: List[str],
    required_docids: Iterable[str] | None = None,
    max_docs: int | None = None,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Load MIRACL Thai corpus from one or more JSONL files, optionally selecting
    only a subset of documents.

    Args:
        json_paths: list of file paths, e.g. ["docs-0.jsonl", "docs-1.jsonl"].
        required_docids: docids that must be present in the returned corpus.
        max_docs: cap on the total number of documents to keep (including
            required_docids). If set, the loader keeps the first docs that fit
            while reserving space for still-unseen required_docids.

    Returns:
        corpus_texts: list[str] of document texts
        docid_to_idx: dict mapping docid -> index in corpus_texts
    """
    required = set(required_docids or [])
    if max_docs is not None:
        if max_docs <= 0:
            raise ValueError("max_docs must be positive when provided")
        if required and len(required) > max_docs:
            raise ValueError(
                "max_docs is smaller than the number of required documents"
            )

    corpus_texts: List[str] = []
    docid_to_idx: Dict[str, int] = {}

    def add_doc(docid: str, full_text: str) -> None:
        docid_to_idx[docid] = len(corpus_texts)
        corpus_texts.append(full_text)

    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)  # each line contains a JSON object (JSONL)
                did = doc.get("docid")
                title = (doc.get("title") or "").strip()
                text = (doc.get("text") or "").strip()
                full_text = f"{title}\n\n{text}" if title and text else title or text
                if required and did in required:
                    add_doc(did, full_text)
                    required.remove(did)
                elif max_docs is None:
                    add_doc(did, full_text)
                else:
                    remaining_capacity = max_docs - len(corpus_texts)
                    required_pending = len(required)
                    if remaining_capacity > required_pending:
                        add_doc(did, full_text)

        if max_docs is not None and len(corpus_texts) >= max_docs and not required:
            break

    if required:
        sample = ", ".join(sorted(required)[:5])
        raise ValueError(
            f"Required docids missing from corpus subset (examples: {sample})"
        )

    return corpus_texts, docid_to_idx


def load_miracl_th(
    corpus: List = ["data/docs-0.jsonl", "data/docs-1.jsonl"],
    topics_path: str = "data/topics.miracl-v1.0-th-dev.tsv",
    qrels_path: str = "data/qrels.miracl-v1.0-th-dev.tsv",
    doc_limit: int | None = None,
) -> Tuple[List[str], Dict[str, List[int]], List[str], List[str]]:
    """
    Returns:
      corpus_texts: list[str]
      qrels: map query_id -> list[doc_index] of relevant docs
      queries: list[str] aligned with query_ids
      query_ids: list[str]
    """
    # 1) Parse topics (qid \t query)
    queries: List[str] = []
    query_ids: List[str] = []
    with open(topics_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            # some HF TSVs can be split strangely; normalize
            if len(row) == 1 and "\t" in row[0]:
                row = row[0].split("\t")
            if not row:
                continue
            qid = row[0].strip()
            query = row[1].strip() if len(row) > 1 else ""
            if qid:
                query_ids.append(qid)
                queries.append(query)

    selected_query_ids = set(query_ids)

    # 2) Parse qrels (qid \t Q0 \t docid \t rel)
    qrels_docids: Dict[str, List[str]] = defaultdict(list)
    required_docids = set()
    with open(qrels_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            # fallback for space-separated variants
            if len(row) == 1:
                row = row[0].split()
            if len(row) < 4:
                continue
            qid, _Q0, docid, rel = (
                row[0].strip(),
                row[1].strip(),
                row[2].strip(),
                row[3].strip(),
            )
            try:
                if int(rel) > 0 and qid in selected_query_ids:
                    qrels_docids[qid].append(docid)
                    required_docids.add(docid)
            except ValueError:
                continue

    # 3) Import corpus with optional downsampling
    corpus_texts, docid_to_idx = load_full_miracl_th_corpus(
        corpus,
        required_docids=required_docids,
        max_docs=doc_limit,
    )

    # 4) Build qrels map using corpus indices
    qrels_map: Dict[str, List[int]] = {}
    missing_docids = []
    for qid in query_ids:
        indices: List[int] = []
        for docid in qrels_docids.get(qid, []):
            try:
                indices.append(docid_to_idx[docid])
            except KeyError:
                missing_docids.append(docid)
        qrels_map[qid] = indices

    if missing_docids:
        sample = ", ".join(sorted(set(missing_docids))[:5])
        raise ValueError(
            f"Some relevant docids were not found in the corpus subset: {sample}"
        )

    return corpus_texts, qrels_map, queries, query_ids


# =========================
# Embedding backends
# =========================
class Embedder:
    def embed_docs(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self.embed_docs(texts)


def _embed_batch_with_retry(
    embed_fn,
    texts: List[str],
    start_idx: int,
    backend_name: str,
    state: Dict[str, int | None],
    failed_indices: List[int],
    failed_doc_behavior: str,
    kept_indices: List[int],
) -> List[List[float]]:
    def _vector_dim(first_vector: object) -> int | None:
        try:
            arr = np.asarray(first_vector, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim == 1 and arr.size > 0:
            return int(arr.shape[0])
        return None

    if not texts:
        return []
    try:
        vectors = embed_fn(texts)
        if vectors and state.get("dim") is None:
            dim = _vector_dim(vectors[0])
            if dim is not None:
                state["dim"] = dim
        kept_indices.extend(range(start_idx, start_idx + len(texts)))
        return vectors
    except Exception as exc:
        if len(texts) == 1:
            original = texts[0]
            candidates = [original]

            # Retry with progressively safer variants for edge-case inputs.
            compact = " ".join(original.split())
            if compact != original:
                candidates.append(compact)
            if len(compact) > 8000:
                candidates.append(compact[:8000])
            if len(compact) > 4000:
                candidates.append(compact[:4000])
            if len(compact) > 2000:
                candidates.append(compact[:2000])
            if len(compact) > 1000:
                candidates.append(compact[:1000])

            seen = set()
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    vectors = embed_fn([candidate])
                    if vectors and state.get("dim") is None:
                        dim = _vector_dim(vectors[0])
                        if dim is not None:
                            state["dim"] = dim
                    kept_indices.append(start_idx)
                    return vectors
                except Exception:
                    continue

            failed_indices.append(start_idx)
            sample = original.replace("\n", " ")[:120]
            if failed_doc_behavior == "skip-doc":
                print(
                    f"WARNING: {backend_name} failed on text index {start_idx}. "
                    f"Skipping doc. Snippet: {sample}"
                )
                return []

            if failed_doc_behavior == "zero-vector":
                # Last-resort fallback: keep run alive by substituting a zero vector.
                if state.get("dim") is None:
                    try:
                        probe = embed_fn(["test"])
                        if probe:
                            dim = _vector_dim(probe[0])
                            if dim is not None:
                                state["dim"] = dim
                    except Exception:
                        pass
                if state.get("dim") is not None:
                    kept_indices.append(start_idx)
                    print(
                        f"WARNING: {backend_name} failed on text index {start_idx}. "
                        f"Using zero-vector fallback. Snippet: {sample}"
                    )
                    return [[0.0] * int(state["dim"])]

            raise RuntimeError(
                f"{backend_name} failed to embed text at index {start_idx}: {sample}"
            ) from exc
        mid = len(texts) // 2
        left = _embed_batch_with_retry(
            embed_fn,
            texts[:mid],
            start_idx,
            backend_name,
            state,
            failed_indices,
            failed_doc_behavior,
            kept_indices,
        )
        right = _embed_batch_with_retry(
            embed_fn,
            texts[mid:],
            start_idx + mid,
            backend_name,
            state,
            failed_indices,
            failed_doc_behavior,
            kept_indices,
        )
        return left + right


def embed_texts_in_batches(
    texts: List[str],
    embed_fn,
    batch_size: int,
    backend_name: str,
    failed_doc_behavior: str,
) -> Tuple[np.ndarray, List[int], List[int]]:
    if not texts:
        return np.empty((0, 0), dtype=np.float32), [], []

    vectors: List[List[float]] = []
    state: Dict[str, int | None] = {"dim": None}
    failed_indices: List[int] = []
    kept_indices: List[int] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        chunk_vectors = _embed_batch_with_retry(
            embed_fn,
            chunk,
            start,
            backend_name,
            state,
            failed_indices,
            failed_doc_behavior,
            kept_indices,
        )
        vectors.extend(chunk_vectors)

    if vectors:
        arr = np.array(vectors, dtype=np.float32)
    else:
        dim = int(state["dim"]) if state["dim"] is not None else 0
        arr = np.empty((0, dim), dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(
            f"{backend_name} returned unexpected embedding shape: {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"{backend_name} produced non-finite values (NaN/Inf)."
        )
    if failed_indices:
        print(
            f"WARNING: {backend_name} had failures for "
            f"{len(failed_indices)} texts."
        )
    return arr, kept_indices, failed_indices


def append_failed_docs_md(
    path: Path,
    model_name: str,
    failed_indices: List[int],
    corpus_texts: List[str],
    behavior: str,
) -> None:
    if not failed_indices:
        return
    lines: List[str] = []
    if not path.exists():
        lines.extend(
            [
                "# Failed Embedding Docs",
                "",
                "Documents that failed during embedding and required fallback handling.",
                "",
            ]
        )

    lines.extend(
        [
            f"## {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | model={model_name} | behavior={behavior}",
            "",
        ]
    )
    action = "Skipping doc" if behavior == "skip-doc" else "Using zero-vector fallback"
    for idx in failed_indices:
        snippet = corpus_texts[idx].replace("\n", " ")[:120]
        lines.append(
            f"- WARNING: {model_name} failed on text index {idx}. {action}. Snippet: {snippet}"
        )
    lines.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base_url: str):
        self.inner = OllamaEmbeddings(model=model, base_url=base_url, keep_alive=0)
        self.name = f"ollama:{model}"

    def embed_docs_with_status(
        self, texts: List[str], failed_doc_behavior: str
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        return embed_texts_in_batches(
            texts=texts,
            embed_fn=self.inner.embed_documents,
            batch_size=OLLAMA_EMBED_BATCH_SIZE,
            backend_name=self.name,
            failed_doc_behavior=failed_doc_behavior,
        )

    def embed_docs(self, texts: List[str]) -> np.ndarray:
        arr, _, _ = self.embed_docs_with_status(
            texts=texts, failed_doc_behavior="zero-vector"
        )
        return arr


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str):
        if not HAS_OPENAI:
            raise RuntimeError(
                "OpenAI backend not installed. pip install langchain-openai openai"
            )
        self.inner = OpenAIEmbeddings(model=model)
        self.name = f"openai:{model}"

    def embed_docs_with_status(
        self, texts: List[str], failed_doc_behavior: str
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        return embed_texts_in_batches(
            texts=texts,
            embed_fn=self.inner.embed_documents,
            batch_size=OPENAI_EMBED_BATCH_SIZE,
            backend_name=self.name,
            failed_doc_behavior=failed_doc_behavior,
        )

    def embed_docs(self, texts: List[str]) -> np.ndarray:
        arr, _, _ = self.embed_docs_with_status(
            texts=texts, failed_doc_behavior="zero-vector"
        )
        return arr


# =========================
# Similarity + evaluation
# =========================
def l2_normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / norms


def cosine_topk(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int):
    q = l2_normalize(query_vecs)
    d = l2_normalize(doc_vecs)
    sims = q @ d.T  # (Q, D)
    k_eff = min(k, d.shape[0])
    idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
    idx = np.take_along_axis(
        idx, np.argsort(-np.take_along_axis(sims, idx, axis=1)), axis=1
    )
    return idx, sims


def evaluate_ranking(
    topk_idx: np.ndarray, qrels: Dict[str, List[int]], query_ids: List[str], k_eval: int
):
    first_ranks, gains_lists = [], []
    for qi, qid in enumerate(query_ids):
        rel = set(qrels.get(qid, []))
        ranked = topk_idx[qi].tolist()
        gains_lists.append([1 if didx in rel else 0 for didx in ranked])

        rank = math.inf
        for pos, didx in enumerate(ranked, start=1):
            if didx in rel:
                rank = pos
                break
        first_ranks.append(rank)

    r = recall_at_k(first_ranks, k_eval)
    mrr = mrr_at_k(first_ranks, k_eval)
    ndcg = ndcg_at_k(gains_lists, k_eval)
    return r, mrr, ndcg


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if value.lower() == "none":
        return None
    return int(value)


def _format_optional_int(value: int | None) -> str:
    return "None" if value is None else str(value)


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def _doc_embed_cache_path(
    model_name: str, query_limit: int, doc_limit: int | None
) -> Path:
    query_part = str(query_limit)
    doc_part = _format_optional_int(doc_limit)
    use_title = "titletext1" if USE_TITLE_PLUS_TEXT else "titletext0"
    filename = (
        f"{_sanitize_name(model_name)}__q{query_part}__d{doc_part}__{use_title}.npy"
    )
    return DOC_EMBED_CACHE_DIR / filename


def _doc_embed_index_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".idx.json")


def _parse_models_arg(models_arg: str | None) -> List[str]:
    aliases = {
        "qwen3:0.6b": "qwen3-embedding:0.6b",
    }

    def normalize_model_name(name: str) -> str:
        return aliases.get(name, name)

    if not models_arg:
        return [normalize_model_name(m) for m in MODELS]
    parsed = [normalize_model_name(m.strip()) for m in models_arg.split(",") if m.strip()]
    if not parsed:
        raise ValueError("No valid model names were provided in --models")
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(parsed))


def _resolve_effective_query_limit(
    topics_path: str = "data/topics.miracl-v1.0-th-dev.tsv",
) -> int:
    total = 0
    with open(topics_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if len(row) == 1 and "\t" in row[0]:
                row = row[0].split("\t")
            if not row:
                continue
            qid = row[0].strip()
            if qid:
                total += 1
    return total


def _format_optional_value(value: object) -> str:
    if value is None:
        return "Unknown"
    return str(value)


def fetch_ollama_model_metadata(model_name: str) -> Dict[str, object]:
    if model_name.startswith("openai:"):
        return {"parameters": None}

    try:
        payload = json.dumps({"model": model_name}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {"parameters": None}

    details = data.get("details", {}) if isinstance(data, dict) else {}
    model_info = data.get("model_info", {}) if isinstance(data, dict) else {}

    parameters = details.get("parameter_size")
    if parameters is None:
        param_count = model_info.get("general.parameter_count")
        parameters = str(param_count) if param_count is not None else None

    return {"parameters": parameters}


def infer_embedding_dim_from_doc_cache(
    model_name: str, query_limit: int, doc_limit: int | None
) -> int | None:
    cache_path = _doc_embed_cache_path(
        model_name, query_limit=query_limit, doc_limit=doc_limit
    )
    if not cache_path.exists():
        return None
    try:
        arr = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None
    if len(arr.shape) == 2:
        return int(arr.shape[1])
    return None


def load_results_md(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [line for line in lines if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return []

    headers = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
    idx = {name: i for i, name in enumerate(headers)}
    required = [
        "Model",
        "Corpus Doc Limit",
        "Top K",
        "Recall",
        "MRR",
        "nDCG",
    ]
    if any(col not in idx for col in required):
        return []

    entries: List[Dict[str, object]] = []
    for row in table_lines[2:]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if len(cells) < len(headers):
            continue

        def get_value(col: str) -> str:
            return cells[idx[col]].strip()

        embedding_dim_raw = get_value("Embedding Dim") if "Embedding Dim" in idx else ""
        parameters_raw = get_value("Parameters") if "Parameters" in idx else ""
        embedding_dim = None
        if embedding_dim_raw and embedding_dim_raw.lower() != "unknown":
            try:
                embedding_dim = int(embedding_dim_raw)
            except ValueError:
                embedding_dim = None

        parameters = (
            None
            if not parameters_raw or parameters_raw.lower() == "unknown"
            else parameters_raw
        )
        try:
            entries.append(
                {
                    "model": get_value("Model"),
                    "doc_limit": _parse_optional_int(get_value("Corpus Doc Limit")),
                    "k": int(get_value("Top K")),
                    "embedding_dim": embedding_dim,
                    "parameters": parameters,
                    "recall": float(get_value("Recall")),
                    "mrr": float(get_value("MRR")),
                    "ndcg": float(get_value("nDCG")),
                }
            )
        except ValueError:
            continue
    return entries


def save_results_md(path: Path, entries: List[Dict[str, object]]) -> None:
    total_queries = _resolve_effective_query_limit()
    sorted_entries = sorted(
        entries,
        key=lambda e: (
            str(e["model"]),
            -1 if e["doc_limit"] is None else int(e["doc_limit"]),
            int(e["k"]),
        ),
    )
    lines = [
        "# Embedding Evaluation Results",
        "",
        f"Total queries: {total_queries} (from data/topics.miracl-v1.0-th-dev.tsv)",
        "",
        "| Model | Corpus Doc Limit | Top K | Embedding Dim | Parameters | Recall | MRR | nDCG |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for entry in sorted_entries:
        lines.append(
            "| "
            f"{entry['model']} | "
            f"{_format_optional_int(entry['doc_limit'])} | "
            f"{entry['k']} | "
            f"{_format_optional_value(entry.get('embedding_dim'))} | "
            f"{_format_optional_value(entry.get('parameters'))} | "
            f"{entry['recall']:.6f} | "
            f"{entry['mrr']:.6f} | "
            f"{entry['ndcg']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remap_qrels_for_kept_docs(
    qrels_map: Dict[str, List[int]], kept_doc_indices: List[int]
) -> Dict[str, List[int]]:
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_doc_indices)}
    remapped: Dict[str, List[int]] = {}
    for qid, rel_doc_indices in qrels_map.items():
        remapped[qid] = [old_to_new[d] for d in rel_doc_indices if d in old_to_new]
    return remapped


# =========================
# Run (no CLI)
# =========================
def run(
    doc_limit: int | None = DOC_LIMIT_DEFAULT,
    k: int = K,
    models: List[str] | None = None,
    force: bool = False,
    failed_doc_behavior: str = FAILED_DOC_BEHAVIOR,
):
    effective_query_limit = _resolve_effective_query_limit()

    selected_models = models if models is not None else MODELS
    existing_entries = load_results_md(RESULTS_MD_PATH)
    cache: Dict[Tuple[str, int | None, int], Dict[str, object]] = {
        (str(e["model"]), e["doc_limit"], int(e["k"])): e
        for e in existing_entries
    }

    models_to_eval: List[str] = []
    model_metrics: List[Tuple[str, float, float, float]] = []
    for model_name in selected_models:
        key = (model_name, doc_limit, k)
        if key in cache and not force:
            entry = cache[key]
            if entry.get("embedding_dim") is None:
                cached_dim = infer_embedding_dim_from_doc_cache(
                    model_name, query_limit=effective_query_limit, doc_limit=doc_limit
                )
                if cached_dim is not None:
                    entry["embedding_dim"] = cached_dim
            if entry.get("parameters") is None:
                meta = fetch_ollama_model_metadata(model_name)
                if (
                    entry.get("parameters") is None
                    and meta.get("parameters") is not None
                ):
                    entry["parameters"] = meta["parameters"]
            recall = float(entry["recall"])
            mrr = float(entry["mrr"])
            ndcg = float(entry["ndcg"])
            model_metrics.append((model_name, recall, mrr, ndcg))
            print(
                f"\n== Model: {model_name} ==\n"
                f"Using cached result from {RESULTS_MD_PATH}: "
                f"dim={_format_optional_value(entry.get('embedding_dim'))}  "
                f"params={_format_optional_value(entry.get('parameters'))}  "
                f"@K={k}  Recall={recall:.4f}  MRR={mrr:.4f}  nDCG={ndcg:.4f}"
            )
        else:
            models_to_eval.append(model_name)

    if models_to_eval:
        # Build combined dataset only when at least one model needs evaluation.
        corpus_texts, qrels_map, queries, query_ids = load_miracl_th(
            doc_limit=doc_limit,
        )
        print(
            f"Loaded MIRACL-TH combined | docs={len(corpus_texts)} | "
            f"queries={len(queries)} | K={k}"
        )
        DOC_EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Prepare embedders for models that are not in cache.
        embedders = []
        for model_name in models_to_eval:
            if model_name.startswith("openai:"):
                embedders.append(
                    (model_name, OpenAIEmbedder(model_name.split("openai:", 1)[1]))
                )
            else:
                embedders.append(
                    (model_name, OllamaEmbedder(model_name, base_url=OLLAMA_BASE_URL))
                )

        for name, emb in embedders:
            print(f"\n== Model: {name} ==")
            cache_path = _doc_embed_cache_path(
                name, query_limit=effective_query_limit, doc_limit=doc_limit
            )
            index_path = _doc_embed_index_path(cache_path)
            kept_doc_indices: List[int] | None = None
            failed_doc_indices: List[int] = []
            if cache_path.exists():
                doc_vecs = np.load(cache_path, allow_pickle=False)
                if index_path.exists():
                    try:
                        idx_payload = json.loads(index_path.read_text(encoding="utf-8"))
                        if isinstance(idx_payload, list) and all(
                            isinstance(x, int) for x in idx_payload
                        ):
                            kept_doc_indices = idx_payload
                    except Exception:
                        kept_doc_indices = None
                if kept_doc_indices is None and doc_vecs.shape[0] == len(corpus_texts):
                    kept_doc_indices = list(range(len(corpus_texts)))
                if kept_doc_indices is None or doc_vecs.shape[0] != len(kept_doc_indices):
                    print("Cached embeddings/index mismatch; recomputing corpus embeddings...")
                    doc_vecs, kept_doc_indices, failed_doc_indices = emb.embed_docs_with_status(
                        corpus_texts,
                        failed_doc_behavior=failed_doc_behavior,
                    )
                    np.save(cache_path, doc_vecs)
                    index_path.write_text(json.dumps(kept_doc_indices), encoding="utf-8")
                else:
                    print(f"Using cached doc embeddings: {cache_path}")
                    if len(kept_doc_indices) < len(corpus_texts):
                        kept_set = set(kept_doc_indices)
                        failed_doc_indices = [
                            i for i in range(len(corpus_texts)) if i not in kept_set
                        ]
            else:
                print("Embedding corpus...")
                doc_vecs, kept_doc_indices, failed_doc_indices = emb.embed_docs_with_status(
                    corpus_texts,
                    failed_doc_behavior=failed_doc_behavior,
                )
                np.save(cache_path, doc_vecs)
                index_path.write_text(json.dumps(kept_doc_indices), encoding="utf-8")
                print(f"Saved doc embeddings cache: {cache_path}")

            if kept_doc_indices is None:
                kept_doc_indices = list(range(len(corpus_texts)))
            if failed_doc_indices:
                append_failed_docs_md(
                    FAILED_DOCS_MD_PATH,
                    model_name=name,
                    failed_indices=failed_doc_indices,
                    corpus_texts=corpus_texts,
                    behavior=failed_doc_behavior,
                )

            eval_corpus_texts = [corpus_texts[i] for i in kept_doc_indices]
            eval_qrels_map = remap_qrels_for_kept_docs(qrels_map, kept_doc_indices)
            dropped_docs = len(corpus_texts) - len(kept_doc_indices)
            if dropped_docs > 0:
                print(
                    f"Skipped {dropped_docs} docs for model {name} due to embedding failures."
                )

            print("Embedding queries...")
            query_vecs = emb.embed_queries(queries)
            embedding_dim = (
                int(query_vecs.shape[1]) if len(query_vecs.shape) == 2 else None
            )
            model_meta = fetch_ollama_model_metadata(name)

            print("Ranking + metrics...")
            topk_idx, _ = cosine_topk(query_vecs, doc_vecs, k=k)
            r, mrr, ndcg = evaluate_ranking(topk_idx, eval_qrels_map, query_ids, k_eval=k)
            model_metrics.append((name, r, mrr, ndcg))
            cache[(name, doc_limit, k)] = {
                "model": name,
                "doc_limit": doc_limit,
                "k": k,
                "embedding_dim": embedding_dim,
                "parameters": model_meta.get("parameters"),
                "recall": r,
                "mrr": mrr,
                "ndcg": ndcg,
            }
            print(
                f"dim={_format_optional_value(embedding_dim)}  "
                f"params={_format_optional_value(model_meta.get('parameters'))}  "
                f"@K={k}  Recall={r:.4f}  MRR={mrr:.4f}  nDCG={ndcg:.4f}"
            )

            # A few qualitative samples
            for qi in range(min(3, len(queries))):
                print(f"\nQuery: {queries[qi][:120]}")
                for rank, didx in enumerate(topk_idx[qi][: min(3, k)], start=1):
                    snippet = eval_corpus_texts[didx].replace("\n", " ")
                    print(f"  {rank}. doc#{didx}: {snippet[:150]}...")
    else:
        print(
            "All selected models already have cached results for "
            f"doc_limit={doc_limit}, k={k}."
        )

    # Keep output order aligned with MODELS and persist full cache.
    model_metrics_by_name = {
        name: (name, r, mrr, ndcg) for name, r, mrr, ndcg in model_metrics
    }
    model_metrics = [
        model_metrics_by_name[name]
        for name in selected_models
        if name in model_metrics_by_name
    ]
    save_results_md(RESULTS_MD_PATH, list(cache.values()))

    if model_metrics:
        best_recall = max(m[1] for m in model_metrics)
        print("\n== Recall Summary ==")
        for name, recall, mrr, ndcg in model_metrics:
            marker = " <= best" if recall == best_recall else ""
            print(f"{name:<30} Recall@{k}: {recall:.4f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare embedding models on MIRACL TH"
    )
    parser.add_argument(
        "--doc-limit",
        type=int,
        default=DOC_LIMIT_DEFAULT,
        help="Maximum number of corpus documents to keep (default: %(default)s).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="Top-K for retrieval metrics (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names. Defaults to MODELS in code.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate selected models even when matching cached metrics exist.",
    )
    parser.add_argument(
        "--failed-doc-behavior",
        type=str,
        default=FAILED_DOC_BEHAVIOR,
        choices=["skip-doc", "zero-vector", "fail"],
        help="How to handle texts that fail embedding.",
    )
    args = parser.parse_args()

    run(
        doc_limit=args.doc_limit,
        k=args.k,
        models=_parse_models_arg(args.models),
        force=args.force,
        failed_doc_behavior=args.failed_doc_behavior,
    )
