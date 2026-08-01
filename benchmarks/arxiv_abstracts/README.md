# W2 — arXiv Abstract Triage

**Primary operators:** `sem_filter` + `sem_extract` + `sem_rank`

## Source
- **Dataset:** arXiv metadata + abstracts (Kaggle mirror).
- **Subset:** 10K cs.LG abstracts.
- **License:** arXiv metadata is CC0; abstracts per arXiv terms — cite arXiv.

## Task
1. `sem_filter` to "papers about learned indexes".
2. `sem_extract` `{contribution_type, evaluation_dataset}`.
3. `sem_rank` top-20 by "most likely to be a systems contribution".

## Ground truth
- Manual labels on a held-out 200-row set.

## Preprocessing
- Fixed 80/10/10 splits, seed 42; parquet under `splits/`.
- Test set untouched until the final run.

## Status
Loader stub in `benchmarks/loader.py::load_arxiv_abstracts`.
