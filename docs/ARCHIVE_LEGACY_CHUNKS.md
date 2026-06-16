# Archiving legacy chunk outputs

Before re-running Phase B baseline evaluation, archive the invalid `chunks_val/` directory.

## Why archive?

The original chunked evaluation had a bug: every chunk processed samples 0–49 instead of distinct slices. All 10 chunk files contain the **same 50 question_ids**, not 500 distinct questions.

## Steps (on Kaggle or local)

```bash
# From project root
mkdir -p data/outputs/_archive_chunks_val_duplicate50
mv data/outputs/chunks_val data/outputs/_archive_chunks_val_duplicate50/chunks_val 2>/dev/null || true
mv data/outputs/chunked_progress_val.json data/outputs/_archive_chunks_val_duplicate50/ 2>/dev/null || true
```

## New output locations

| Mode | Directory |
|------|-----------|
| Baseline (`vision_only`) | `data/outputs/baseline_200/` |
| Adaptive (`ocr_adaptive`) | `data/outputs/ocr_adaptive_200/` |
| Comparison reports | `data/outputs/comparisons/` |
| Eval subset | `data/outputs/eval_subset_200.json` |

Do not resume from `chunked_progress_val.json` — each mode uses its own `chunked_progress.json` inside the mode directory.
