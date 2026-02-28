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

The evaluator always uses all available queries in `topics.miracl-v1.0-th-dev.tsv`
(currently 733 queries) and loads a corpus subset controlled by `--doc-limit`
while always preserving all relevant qrels docs for those queries.

### Dependencies

This project uses `uv` for Python environment and dependency management.
Install dependencies with:

```bash
uv sync
```

You need running inference endpoints for the models you enable (for example,
Ollama at `http://localhost:11434` for `embeddinggemma`).

### Usage

Run with configured default models:

```bash
uv run python embedding_compare.py
```

Run specific models:

```bash
uv run python embedding_compare.py --models "bge-m3,qwen3:0.6b" --doc-limit 5000 --k 10
```

CLI flags:

- `--doc-limit` – maximum number of corpus docs to keep (default: `5000`).
- `--k` – ranking depth for metrics (default: `10`).
- `--models` – comma-separated model list.
- `--force` – ignore cached metrics in `result.md` and recompute.
- `--failed-doc-behavior` – one of `skip-doc` (default), `zero-vector`, `fail`.

### Outputs

- `result.md`
  - Written/updated after each run.
  - Includes total query count at top.
  - Contains a markdown table with model metrics and metadata.
- `failed_doc.md`
  - Written when docs fail embedding.
  - Logs model name, failed doc index, and snippet.
- `.cache/doc_embeddings/*.npy` and `*.idx.json`
  - Cached doc embeddings and kept-doc indices to speed up reruns.

### How Query-Doc Mapping Works

Each topic row has `qid`, qrels maps `qid` to relevant `docid` values (`rel > 0`),
and those `docid` values are retrieved from `docs-*.jsonl`.

Examples from this dataset:

1. Query `qid=4`
   - Query: `บันทึกเหตุการณ์และเรื่องราวต่าง ๆ ในยุคสามก๊กฉบับแรก ที่มีการบันทึกเป็นลายลักษณ์อักษรเรียกว่าอะไร?`
   - Relevant doc from qrels: `9800#4`
   - Doc title: `สามก๊ก`

2. Query `qid=7`
   - Query: `เทียรี่ เมฆวัฒนา สามารถเล่นกีต้าร์ได้ใช่หรือไม่?`
   - Relevant doc from qrels: `56155#7`
   - Doc title: `เทียรี่ เมฆวัฒนา`

3. Query `qid=9`
   - Query: `ดิ อะเมซิ่ง เรซ เป็นเรียลลิตี้โชว์จากประเทศอะไร?`
   - Relevant doc from qrels: `202175#0`
   - Doc title: `ดิอะเมซิ่งเรซเอ็นดิสคัฟเวอรีแชนแนล`

4. Query `qid=20`
   - Query: `คอเคลียเต็มไปด้วยน้ำที่เรียกว่าอะไร?`
   - Relevant doc from qrels: `844976#6`
   - Doc title: `หูชั้นในรูปหอยโข่ง`
