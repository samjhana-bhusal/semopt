# W1 — Product Review Complaint Classification

**Primary operator:** `sem_filter`

## Source
- **Dataset:** Amazon Reviews 2023 (McAuley Lab). Publicly available.
- **Subset:** 10K reviews from the Electronics category (5-core).
- **License:** research use per the McAuley Lab terms — cite the dataset paper.

## Task
`sem_filter("Does this review complain about shipping or packaging?")`

## Ground truth
- Hand-label a 500-row test set (crowdsourced labels acceptable if available).

## Preprocessing
- Sample 10K reviews with seed 42.
- Fixed 80/10/10 train/cal/test splits, seed 42; store as parquet under `splits/`.
- **Never touch the test set until the final Pareto run** (SRS §7 data hygiene).

## Status
Loader stub in `benchmarks/loader.py::load_product_reviews` (not wired here — needs
network + a real model backend; gated on API budget, NFR-4).
