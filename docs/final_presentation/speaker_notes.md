# VisionDocPhi-3.5 Final Presentation Speaker Notes

## 10-minute timing

| Slide | Target time | Key phrase |
|-------|-------------|------------|
| 1 | 0:30 | 30% relative ANLS gain without fine-tuning |
| 2 | 1:00 | ~1334 px effective cap from 336 px HD tiles |
| 3 | 1:00 | Wrong field, not wrong model |
| 4 | 0:45 | OCR guides vision, never replaces it |
| 5 | 1:30 | Walk the architecture top-to-bottom |
| 6 | 1:30 | Safe fallback at every stage |
| 7 | 1:00 | Three signals: type, keywords, density |
| 8 | 1:00 | Paired comparison, same 200 questions |
| 9 | 1:00 | Table/list +0.395 — where OCR matters most |
| 10 | 1:00 | Specimen success, Eastern fallback, date failure |
| 11 | 0:30 | System design unlocks SLM capability |

## Short opening

I worked on SmallTalk Docs, where the goal was to see whether a small vision-language model, Phi-3.5-Vision at 4.2B parameters, can reliably answer questions over real documents. My contribution was an adaptive OCR-infused prompting pipeline. Instead of always giving OCR or never giving OCR, it selectively injects extracted text with bounding boxes when the question and document look structured. On 200 paired DocVQA samples, this improved mean ANLS from 0.506 to 0.658 without any fine-tuning.

## Slide 5 architecture explanation

Both paths use the same Phi-3.5-Vision model. The baseline sends only image and question. The adaptive path runs a preprocessing pipeline first. The router decides whether OCR hints are worth trying. If yes, retrieval selects question-relevant OCR lines, spatial expansion adds nearby label-value lines, the formatter cleans and orders text with coordinates, and the quality gate accepts or rejects the snippet. If rejected, the system falls back to the exact baseline input.

## Slide 10 case study script

The specimen case shows the main success pattern: retrieval finds the label, but the answer is on the next line, so spatial expansion adds it. Eastern Airlines shows why the gate matters: the retrieval score was too low, so the system avoided weak OCR and vision-only succeeded. The date case is the honest limitation: Azure OCR misread the handwritten year, and the model copied it. The gate checks confidence, not OCR truth.
