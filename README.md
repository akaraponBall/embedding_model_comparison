## Embedding Comparison Script

`embedding_compare.py` benchmarks embedding models on the MIRACL Thai dev set.
The script loads the Thai corpus, encodes documents and queries, ranks with
cosine similarity, and reports Recall@K, MRR@K, and nDCG@K. A side-by-side
summary highlights the best recall across all configured models.

### Data Files

Place the MIRACL Thai dev resources under `data/` (paths are configurable in the
script). You can download the corpus from the official Hugging Face release:
[miracl/miracl-corpus](https://huggingface.co/datasets/miracl/miracl-corpus/tree/main)

- `docs-*.jsonl` – document shards in JSON Lines format with `docid`, `title`,
  and `text`.
- `topics.miracl-v1.0-th-dev.tsv` – queries (`qid \t query`).
- `qrels.miracl-v1.0-th-dev.tsv` – relevance judgments (`qid \t Q0 \t docid \t rel`).

The loader keeps a subset of the corpus when `CORPUS_DOC_LIMIT` is set, but it
always retains every document referenced by the selected qrels.

### Dependencies

Install the Python dependencies before running the script:

```bash
pip install -r requirements.txt
# or, if using uv/pdm/poetry, install according to your environment
```

You need running inference endpoints for the models you enable (for example,
Ollama at `http://localhost:11434` for `embeddinggemma`).

### Usage

Run the script directly to evaluate the configured models:

```bash
python embedding_compare.py
```

CLI flags override common settings:

- `--limit` – number of queries to evaluate (default matches `LIMIT`).
- `--k` – ranking depth for metrics (default matches `K`).

Adjust constants in the top of the script to add/remove models, change the
Ollama base URL, or cap the corpus via `CORPUS_DOC_LIMIT` when working on
resource-constrained machines.
