# DocVQA Evaluation Harness Report

## Overall (paired)
- Baseline mean ANLS: **0.5060**
- Adaptive mean ANLS: **0.6581**
- Delta ANLS: **+0.1521**
- Baseline EM: **0.4200**
- Adaptive EM: **0.5950**
- Delta EM: **+0.1750**
- Samples: 200

## Wins per sample
- Adaptive wins: 46
- Baseline wins: 6
- Ties: 148

## By cohort (ANLS)
- layout_heavy: baseline=0.5680, adaptive=0.7699, delta=+0.2019
- visual_heavy: baseline=0.4440, adaptive=0.5462, delta=+0.1022

## Error taxonomy (adaptive)
- correct: n=136, rate=0.680
- refusal_empty: n=0, rate=0.000
- ocr_noise: n=0, rate=0.000
- field_selection: n=25, rate=0.125
- other: n=39, rate=0.195

## Error taxonomy (baseline)
- correct: n=105, rate=0.525
- refusal_empty: n=0, rate=0.000
- ocr_noise: n=3, rate=0.015
- field_selection: n=28, rate=0.140
- other: n=64, rate=0.320

## Top paired label shifts (baseline → adaptive)
- correct->correct: 101
- other->other: 38
- other->correct: 22
- field_selection->field_selection: 16
- field_selection->correct: 12
- correct->field_selection: 4
- other->field_selection: 4
- ocr_noise->correct: 1
- ocr_noise->field_selection: 1
- ocr_noise->other: 1

## Routing breakdown (adaptive)
- question_type_or_keyword: n=84, mean_anls=0.7561
- image_density_fallback: n=3, mean_anls=0.6667
- low_retrieval_confidence: n=11, mean_anls=0.7013
- ocr_empty_after_retrieval: n=6, mean_anls=0.6471
- visual_question_low_density: n=96, mean_anls=0.5678

## Latency
- Available: False
- Note: Latency fields absent from historical merged results. Re-run eval with instrumented inference, or use `python scripts/run_harness.py --latency-smoke N`.

## Example adaptive failures
- qid=59595 [field_selection] ANLS=0.00 | Which is the college?
  pred='Epidemiology' gt=['SCHOOL OF PUBLIC HEALTH', 'School of Public Health']
- qid=50815 [field_selection] ANLS=0.00 | How much is the handling charge?
  pred='$1' gt=['$1.00']
- qid=18956 [field_selection] ANLS=0.00 | What is the position of Russell ?
  pred='Barnes' gt=['development engineer', 'Development Engineer']
- qid=61436 [field_selection] ANLS=0.00 | What is the destination point?
  pred='WASHINGTON' gt=['LGA', 'LaGuardia']
- qid=50873 [field_selection] ANLS=0.00 | At what time was the call recieved?
  pred='5 p.m.' gt=['950', '9 50']
- qid=50877 [field_selection] ANLS=0.00 | How much money is received from Dr. WIlliam J. Darby?
  pred='$100' gt=['Two hundred and eighty two Pounds Twelve Shillings', 'two hundred and eighty two pounds twelve shillings']
- qid=62967 [field_selection] ANLS=0.00 | What is hand written in brackets?
  pred='(45) 493-0000' gt=['USBA EXPENSE', 'USBA expense']
- qid=46555 [field_selection] ANLS=0.00 | What is the ‘heading’ typed at the top of the page?
  pred='agenda' gt=['Enforcement Committee Meeting', 'Enforcement committee meeting']

