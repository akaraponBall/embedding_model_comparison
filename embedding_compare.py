# =========================
# Imports
# =========================
import argparse
import math
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

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
LIMIT = 200                          # number of queries to evaluate (slice, for speed)
CORPUS_DOC_LIMIT = LIMIT * 10       # max docs to keep in corpus (None = use full corpus)
K = 10                              # @K for metrics
OLLAMA_BASE_URL = "http://localhost:11434"
MODELS = [
    "embeddinggemma",
    "qwen3-embedding:0.6b",
    "bge-large",
    "nomic-embed-text",
    "mxbai-embed-large"
    # "openai:text-embedding-3-small",  # requires OPENAI_API_KEY + extra deps
]
USE_TITLE_PLUS_TEXT = False   # set True to concatenate "title: text" as the doc string

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
                doc = json.loads(line)   # each line contains a JSON object (JSONL)
                did = doc.get("docid")
                title = (doc.get("title") or "").strip()
                text  = (doc.get("text") or "").strip()
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

def load_miracl_th(limit: int = None,
                   corpus: List = ['data/docs-0.jsonl', 'data/docs-1.jsonl'],
                   topics_path: str = 'data/topics.miracl-v1.0-th-dev.tsv',
                   qrels_path: str = 'data/qrels.miracl-v1.0-th-dev.tsv',
                   doc_limit: int | None = None) -> Tuple[List[str], Dict[str, List[int]], List[str], List[str]]:
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

    if limit is not None and limit < len(query_ids):
        query_ids = query_ids[:limit]
        queries   = queries[:limit]

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
            qid, _Q0, docid, rel = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
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
            "Some relevant docids were not found in the corpus subset: "
            f"{sample}"
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

class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base_url: str):
        self.inner = OllamaEmbeddings(model=model, base_url=base_url)
        self.name = f"ollama:{model}"
    def embed_docs(self, texts: List[str]) -> np.ndarray:
        return np.array(self.inner.embed_documents(texts), dtype=np.float32)

class OpenAIEmbedder(Embedder):
    def __init__(self, model: str):
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI backend not installed. pip install langchain-openai openai")
        self.inner = OpenAIEmbeddings(model=model)
        self.name = f"openai:{model}"
    def embed_docs(self, texts: List[str]) -> np.ndarray:
        return np.array(self.inner.embed_documents(texts), dtype=np.float32)

# =========================
# Similarity + evaluation
# =========================
def l2_normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / norms

def cosine_topk(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int):
    q = l2_normalize(query_vecs)
    d = l2_normalize(doc_vecs)
    sims = q @ d.T                              # (Q, D)
    k_eff = min(k, d.shape[0])
    idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
    idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(sims, idx, axis=1)), axis=1)
    return idx, sims

def evaluate_ranking(topk_idx: np.ndarray,
                     qrels: Dict[str, List[int]],
                     query_ids: List[str],
                     k_eval: int):
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


# =========================
# Run (no CLI)
# =========================
def run(limit: int = LIMIT, k: int = K, doc_limit: int | None = CORPUS_DOC_LIMIT):
    # Build combined dataset
    corpus_texts, qrels_map, queries, query_ids = load_miracl_th(
        limit=limit,
        doc_limit=doc_limit,
    )

    print(
        f"Loaded MIRACL-TH combined | docs={len(corpus_texts)} | "
        f"queries={len(queries)} | K={k}"
    )

    # Prepare embedders
    embedders = []
    for m in MODELS:
        if m.startswith("openai:"):
            embedders.append((m, OpenAIEmbedder(m.split("openai:", 1)[1])))
        else:
            embedders.append((m, OllamaEmbedder(m, base_url=OLLAMA_BASE_URL)))

    # Evaluate
    model_metrics = []
    for name, emb in embedders:
        print(f"\n== Model: {name} ==")
        print("Embedding corpus...")
        doc_vecs = emb.embed_docs(corpus_texts)
        print("Embedding queries...")
        query_vecs = emb.embed_queries(queries)

        print("Ranking + metrics...")
        topk_idx, _ = cosine_topk(query_vecs, doc_vecs, k=k)
        r, mrr, ndcg = evaluate_ranking(topk_idx, qrels_map, query_ids, k_eval=k)
        model_metrics.append((name, r, mrr, ndcg))
        print(f"@K={k}  Recall={r:.4f}  MRR={mrr:.4f}  nDCG={ndcg:.4f}")

        # A few qualitative samples
        for qi in range(min(3, len(queries))):
            print(f"\nQuery: {queries[qi][:120]}")
            for rank, didx in enumerate(topk_idx[qi][:min(3, k)], start=1):
                snippet = corpus_texts[didx].replace("\n", " ")
                print(f"  {rank}. doc#{didx}: {snippet[:150]}...")

    if model_metrics:
        best_recall = max(m[1] for m in model_metrics)
        print("\n== Recall Summary ==")
        for name, recall, mrr, ndcg in model_metrics:
            marker = " <= best" if recall == best_recall else ""
            print(f"{name:<30} Recall@{k}: {recall:.4f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare embedding models on MIRACL TH")
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help="Limit number of queries to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="Top-K for retrieval metrics (default: %(default)s)",
    )
    args = parser.parse_args()

    run(limit=args.limit, k=args.k)
