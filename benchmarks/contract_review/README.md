# W3 — Contract Clause Review

**Primary operators:** `sem_extract` + `sem_map`

## Source
- **Dataset:** CUAD (Contract Understanding Atticus Dataset), publicly released.
- **Size:** 510 contracts, expert-labeled clauses.
- **License:** CUAD is released under CC BY 4.0 — cite the CUAD paper.

## Task
- `sem_extract` clause type + risk category from each paragraph.

## Ground truth
- CUAD's own expert labels (already provided) — no manual labeling needed.

## Preprocessing
- Fixed 80/10/10 splits, seed 42; parquet under `splits/`.
- Test set untouched until the final run.

## Status
Loader stub in `benchmarks/loader.py::load_contract_review`.
