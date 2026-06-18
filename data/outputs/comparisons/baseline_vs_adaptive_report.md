# Baseline vs OCR-Adaptive Comparison

- Baseline mean ANLS: **0.5060**
- Adaptive mean ANLS: **0.6581**
- Delta (adaptive - baseline): **+0.1521**
- Samples: 200

## Wins per sample
- Adaptive wins: 46
- Baseline wins: 6
- Ties: 148

## By cohort
- layout_heavy: baseline=0.5680, adaptive=0.7699, delta=+0.2019
- visual_heavy: baseline=0.4440, adaptive=0.5462, delta=+0.1022

## Routing breakdown (adaptive)
- question_type_or_keyword: n=84, mean_anls=0.7561
- image_density_fallback: n=3, mean_anls=0.6667
- low_retrieval_confidence: n=11, mean_anls=0.7013
- ocr_empty_after_retrieval: n=6, mean_anls=0.6471
- visual_question_low_density: n=96, mean_anls=0.5678
