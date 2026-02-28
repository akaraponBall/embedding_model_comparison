# Embedding Evaluation Results

Total queries: 733 (from data/topics.miracl-v1.0-th-dev.tsv)

| Model | Corpus Doc Limit | Top K | Embedding Dim | Parameters | Recall | MRR | nDCG |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| bge-m3 | 5000 | 5 | 1024 | 566.70M | 0.995907 | 0.963847 | 0.967359 |
| bge-m3 | 80000 | 5 | 1024 | 566.70M | 0.976808 | 0.922010 | 0.927621 |
| qwen3-embedding:0.6b | 5000 | 5 | 1024 | 595.78M | 0.987722 | 0.935130 | 0.943719 |
| qwen3-embedding:latest | 5000 | 5 | 4096 | 7.6B | 0.997271 | 0.960596 | 0.966131 |

## How Query-Doc Mapping Works

The evaluation uses two files together:

- `topics.miracl-v1.0-th-dev.tsv`: query id -> query text
- `qrels.miracl-v1.0-th-dev.tsv`: query id -> relevant document ids

Each qrels row has this format:

`qid    Q0    docid    rel`

Example:

`4    Q0    9800#4    1`

Meaning:

- `4`: query id (look up query text with id `4` in `topics`)
- `Q0`: placeholder field (kept for standard format; not used by scoring logic)
- `9800#4`: document id judged for this query
- `1`: relevance label (`> 0` means relevant/correct in this project)

So for query id `4`, if the model retrieves doc `9800#4`, that retrieval is counted as a relevant hit.
