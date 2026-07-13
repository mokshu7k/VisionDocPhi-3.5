# VisionDocPhi-3.5 Final Presentation Slide Content

**Presentation length:** 10 minutes  
**Theme:** Improving Phi-3.5-Vision for dense document question answering using an adaptive OCR-infused pipeline  
**Main result:** Vision-only mean ANLS **0.506** to OCR-adaptive mean ANLS **0.658** on 200 paired SP-DocVQA samples

---

## Suggested 10-Minute Flow

| Slide | Title | Time |
|---|---|---:|
| 1 | Title and Core Idea | 0:30 |
| 2 | Why Phi-3.5-Vision? | 0:55 |
| 3 | Problem: Dense Document Images | 0:55 |
| 4 | Proposed Solution | 0:55 |
| 5 | Overall Architecture | 1:15 |
| 6 | Adaptive Pipeline Methodology | 1:05 |
| 7 | Router Architecture | 0:50 |
| 8 | Hybrid Retrieval and Spatial Expansion | 0:55 |
| 9 | Formatting, Cleaning, and Quality Gate | 0:55 |
| 10 | Metric: ANLS | 0:50 |
| 11 | Results Comparison | 1:00 |
| 12 | Success, Fallback, and Failure Cases | 1:05 |
| 13 | Conclusion | 0:20 |

---

# Slide 1: Title and Core Idea

**Time:** 0:30

## On-Slide Content

**VisionDocPhi-3.5**  
**Improving Document VQA with Adaptive OCR-Infused Prompting**

SmallTalk Docs: Guiding a small vision-language model through dense documents

**Key message:**  
Adaptive OCR hints improved Phi-3.5-Vision from **0.506 to 0.658 mean ANLS** without fine-tuning.

## Layout Suggestion

Use a clean title slide:

- Left side: project title and one-line subtitle
- Right side: small architecture or document image thumbnail if available
- Bottom: your name, institute, and project context

## Speaker Notes

I worked on improving document question answering using Phi-3.5-Vision, a small multimodal model. The goal was not to fine-tune the model, but to improve how information is presented to it. The main contribution is an adaptive OCR-infused prompting pipeline that selectively gives the model relevant OCR text and spatial coordinates when the document is dense or structured.

---

# Slide 2: Why Phi-3.5-Vision?

**Time:** 0:55

## On-Slide Content

**Phi-3.5-Vision**

- 4.2B parameter multimodal model
- Accepts image + text and generates text answers
- Practical for single-GPU or lower-cost deployment
- Useful for enterprise document QA where cost, latency, and privacy matter

**Limitation for documents:**  
Dense pages contain many small text regions, and the model has limited effective visual resolution.

## Visual Suggestion

Add a simple comparison graphic:

```text
Large VLMs
High accuracy, high cost

Phi-3.5-Vision
Lower cost, on-prem friendly, but weaker on dense document text
```

## Speaker Notes

Phi-3.5-Vision is interesting because it is small enough to be practical, but document understanding is still difficult for it. In dense scanned forms, tables, or invoices, the answer may be a tiny field among many similar fields. Since the model processes images at a limited effective resolution, fine text and handwritten values can be hard to read reliably.

---

# Slide 3: Problem: Dense Document Images

**Time:** 0:55

## On-Slide Content

**Problem**

Phi-3.5-Vision struggles when document pages are:

- Dense with fine text
- Structured as forms or tables
- Filled with similar labels and fields
- Low contrast, scanned, or handwritten

**Resolution challenge:**  
Phi vision processing effectively caps high-resolution detail around **1334 x 1334**, so dense text regions get compressed.

**Resulting failures:**

- Misreads small text or handwritten digits
- Selects the wrong field
- Copies boilerplate instead of the answer

## Visual Suggestion

Use a dense document image or the visual asset:

`docs/final_presentation/assets/dense_form_resolution_visual.png`

## Text Below Visual

Dense documents contain many small text regions. When the page is downscaled or tiled into a limited visual budget, the model may preserve the overall layout but lose exact character-level detail.

## Speaker Notes

The problem is not that Phi-3.5-Vision cannot understand documents at all. The issue is precision. For DocVQA, the answer is often one exact value: a date, a name, an amount, or a field entry. On dense forms, the model can see the page layout but may not read the exact field correctly.

---

# Slide 4: Proposed Solution

**Time:** 0:55

## On-Slide Content

**Solution: Adaptive OCR-Infused Prompting**

Instead of:

- Vision-only: image + question
- Always-OCR: full OCR text for every question

Use:

**Adaptive OCR infusion**

- Always pass the image to Phi-3.5-Vision
- Add OCR text only when it is likely to help
- Retrieve only relevant OCR lines
- Preserve bounding-box coordinates
- Reject weak OCR snippets and fall back to vision-only

## Visual Suggestion

Use a simple two-path diagram:

```text
Image + Question
      |
   Router
   /    \
Vision  OCR-assisted
 only   prompt
   \    /
Phi-3.5-Vision
      |
   Answer
```

## Speaker Notes

The core idea is that OCR should guide the vision model, not replace it. The document image is always included. OCR is treated as a hint, and only a short, relevant, coordinate-aware snippet is added to the prompt when the pipeline is confident it will help.

---

# Slide 5: Overall Architecture

**Time:** 1:15

## On-Slide Content

**Overall Architecture**

Inputs:

- Document image
- User question
- Azure OCR lines with bounding boxes

Processing:

- Router decides vision-only or OCR-assisted path
- Retriever selects relevant OCR lines
- Spatial expansion adds nearby label-value text
- Formatter cleans and orders the snippet
- Quality gate accepts or rejects OCR hints

Output:

- Phi-3.5-Vision generates the final short answer

## Diagram to Insert

Use:

`docs/final_presentation/assets/overall_architecture.png`

If the PNG does not fit cleanly, use:

`docs/final_presentation/assets/overall_architecture.svg`

## Text Below Diagram

The same Phi-3.5-Vision model is used in both paths. The baseline sends only the image and question. The adaptive path adds a preprocessing pipeline that may attach relevant OCR hints before the model generates the final answer.

## Speaker Notes

This architecture compares the baseline and adaptive modes. In baseline mode, Phi-3.5-Vision receives only the document image and question. In adaptive mode, the pipeline first decides whether OCR is useful. If yes, it retrieves and formats only the most relevant OCR lines. If the quality gate rejects the snippet, the system falls back to the exact baseline input.

---

# Slide 6: Adaptive Pipeline Methodology

**Time:** 1:05

## On-Slide Content

**Five-Stage Adaptive Pipeline**

1. **Routing**  
   Decide whether OCR should be attempted.

2. **Hybrid Retrieval**  
   Select question-relevant OCR lines using semantic + keyword matching.

3. **Spatial Expansion**  
   Add nearby lines that may contain the value next to a label.

4. **Formatting and Cleaning**  
   Remove boilerplate and preserve coordinates.

5. **Quality Gate**  
   Accept useful OCR or fall back to vision-only.

## Diagram to Insert

Use:

`docs/final_presentation/assets/adaptive_pipeline.png`

If the PNG does not fit cleanly, use:

`docs/final_presentation/assets/adaptive_pipeline.svg`

## Text Below Diagram

The pipeline is designed as a safe filter. Every stage either improves the OCR snippet or sends the sample back to vision-only inference, preventing weak OCR from being forced into the prompt.

## Speaker Notes

The methodology is adaptive because it does not assume OCR is always useful. The router starts by checking the question and page characteristics. If OCR is attempted, retrieval narrows the full OCR page into a small set of relevant lines. Spatial expansion handles forms where the label and value are adjacent but separate. The quality gate is the final safety check before OCR enters the model prompt.

---

# Slide 7: Router Architecture

**Time:** 0:50

## On-Slide Content

**Routing: When Should OCR Be Used?**

The router uses three signals:

- **Question type metadata**: form, table, layout, handwritten
- **Question keywords**: field, row, column, amount, total, date
- **Image density analysis**: edge density and contrast for dense pages

Routing outcomes:

- OCR-assisted path for structured text questions
- Vision-only path for visual or low-density questions
- Fallback when OCR is unavailable or low confidence

## Diagram to Insert

Use:

`docs/final_presentation/assets/router_architecture.png`

If the PNG does not fit cleanly, use:

`docs/final_presentation/assets/router_architecture.svg`

## Text Below Diagram

The router acts as the first decision point. It sends form-like or text-dense samples to the OCR-assisted path, while visual questions or weak OCR cases remain vision-only.

## Speaker Notes

Routing is important because always injecting OCR can be harmful. For example, a figure or diagram question may not need OCR at all. The router therefore checks whether the question suggests structured text and whether the page appears text-dense. Later, the quality gate can still override the router and reject the OCR snippet.

---

# Slide 8: Hybrid Retrieval and Spatial Expansion

**Time:** 0:55

## On-Slide Content

**Hybrid Retrieval**

Goal: narrow full-page OCR to question-relevant lines.

```text
final_score = 0.7 × semantic_score + 0.3 × keyword_score
```

- **Semantic:** MiniLM embeddings on 3-line context windows
- **Keyword:** exact token / field-name matching (e.g. `Specimen`, `Date`)
- **Filter:** drop lines below 0.20; keep top-K within **25 lines / 1200 chars**

**Spatial Expansion**   

Problem: retrieval often finds the label, not the answer value.

- Add vertical neighbour **below** within **45 px**, same column
- If line is label-only (`Specimen:`), add the value on the next line
- Add same-row value to the **right** (6 px Y-align tolerance)
- Re-sort by reading order before formatting

## Visual Suggestion

Use a simple before/after diagram:

```text
Retrieval only          After spatial expansion
─────────────          ───────────────────────
Specimen:              Specimen:
                        10 rat sera
```

Or reuse:

`docs/final_presentation/assets/specimen_success_visual.png`

## Text Below Visual

Hybrid retrieval selects semantically and lexically relevant OCR lines. Spatial expansion then adds nearby label-value pairs that retrieval alone would miss on dense forms.

## Speaker Notes

Retrieval is the first narrowing step. Semantic similarity helps when the question is phrased differently from the OCR text, while keyword matching catches exact field names like specimen or date. Spatial expansion handles the common form pattern where the label and value are on separate lines or in adjacent columns. The specimen case is the clearest example: retrieval finds `Specimen:`, and expansion adds `10 rat sera` below it.

---

# Slide 9: Formatting, Cleaning, and Quality Gate

**Time:** 0:55

## On-Slide Content

**Formatting and Cleaning**

- Remove boilerplate (`INSTRUCTION`, `NOTICE`, `USER`)
- Normalize punctuation and stray `#` markers
- Drop empty or junk lines
- Sort lines top-to-bottom, left-to-right
- Attach coordinates: `[x1,y1,x2,y2] line text`

**Quality Gate**

Final safety check before OCR enters the prompt.

OCR snippet is **accepted** only if:

- Top retrieval score ≥ **0.45**
- Snippet adds information beyond the question tokens
- For "what is" questions: contains label-value or answer-like content

On **reject:** fall back to vision-only (identical to baseline input).

## Visual Suggestion

Use a simple accept/reject flow:

```text
Formatted OCR snippet
        |
   Quality gate
   /          \
Accept        Reject
  |              |
OCR prompt    Vision-only
```

## Text Below Visual

Formatting keeps the snippet short, ordered, and coordinate-aware. The quality gate prevents weak or redundant OCR from being injected into the model prompt.

## Speaker Notes

Cleaning removes noisy OCR that would distract the model without helping. The formatter preserves spatial order so Phi can relate text regions to the image layout. The quality gate is the last safety layer: even if the router sends a sample down the OCR path, a low-confidence or uninformative snippet is rejected. The Eastern Airlines fallback case in the results slide is a good example of this behaviour working correctly.

---

# Slide 10: Metric: ANLS

**Time:** 0:50

## On-Slide Content

**Evaluation Metric: ANLS**

**ANLS = Average Normalized Levenshtein Similarity**

It measures how close the predicted answer is to the ground truth.

Why ANLS is used:

- Standard metric for DocVQA
- Gives partial credit for near matches
- Better than exact match for dates, names, and formatting variations

Formula idea:

```text
similarity = 1 - edit_distance(prediction, ground_truth) / max_length

if similarity < 0.5:
    ANLS = 0
else:
    ANLS is scaled between 0 and 1
```

## Small Example Table

| Prediction | Ground Truth | ANLS Meaning |
|---|---|---|
| `10 rat sera` | `10 rat sera` | Exact match, high score |
| `8/25/86` | `8/25/88` | Near match, partial credit |
| `xyz` | `8/25/88` | Poor match, zero |

## Speaker Notes

ANLS is useful because document answers are short strings, and a prediction can be almost correct without being exactly identical. For example, one wrong digit in a date should not be treated the same as a completely unrelated answer. This is why ANLS is more informative than exact match alone.

---

# Slide 11: Results Comparison

**Time:** 1:00

## On-Slide Content

**Evaluation Setup**

- Dataset: 200 paired SP-DocVQA validation samples
- Same samples for both systems
- Same Phi-3.5-Vision model
- Baseline: image + question only
- Adaptive: image + question + OCR hints only when gate passes

**Main Results**

| Metric | Vision-Only Baseline | OCR-Adaptive V2 | Improvement |
|---|---:|---:|---:|
| Mean ANLS | 0.506 | **0.658** | **+0.152** |
| Exact match | 0.420 | **0.595** | +0.175 |
| Layout-heavy ANLS | 0.568 | **0.770** | +0.202 |
| Visual-heavy ANLS | 0.444 | **0.546** | +0.102 |

**Win/Tie/Loss**

| Outcome | Count |
|---|---:|
| Adaptive wins | 46 |
| Baseline wins | 6 |
| Ties | 148 |

**Question-Type Highlight**

| Question Type | Baseline ANLS | OCR-Adaptive ANLS | Improvement |
|---|---:|---:|---:|
| Form | 0.603 | **0.790** | +0.187 |
| Table/list | 0.376 | **0.771** | **+0.395** |
| Layout | 0.571 | **0.800** | +0.229 |

## Optional Visual

Create a simple bar chart with two bars:

- Vision-only ANLS: 0.506
- OCR-adaptive ANLS: 0.658

## Text Below Table

The adaptive pipeline improves overall ANLS by **+0.152**, approximately a **30% relative gain**. The largest question-type gain is on table/list questions, where OCR and spatial layout cues matter most.

## Speaker Notes

This was a paired comparison, so both methods answered the same 200 questions. The improvement is strongest where the pipeline was designed to help: layout-heavy documents, especially tables and structured forms. The 148 ties are also important because they show that the pipeline does not change many already-correct cases, while adaptive wins are much more frequent than baseline wins.

---

# Slide 12: Success, Fallback, and Failure Cases

**Time:** 1:05

## On-Slide Content

**Case Study 1: Success**

- Question: `What is the specimen?`
- Retrieval found: `Specimen:`
- Spatial expansion added: `10 rat sera`
- Prediction: `10 rat sera`
- ANLS: **1.0**

**Case Study 2: Safe Fallback**

- Question: `What is the name of the issuing airlines?`
- Retrieval confidence was low
- Quality gate rejected OCR
- Vision-only predicted: `Eastern Airlines`
- ANLS: **1.0**

**Case Study 3: Honest Failure**

- Question: `What is the date on which the request for change was prepared?`
- Ground truth: `8/25/88`
- Baseline: `8/25/86`
- OCR-adaptive: `8/25/15`
- Cause: OCR misread handwritten year

## Visual Suggestion

Use:

`docs/final_presentation/assets/specimen_success_visual.png`

Optional second visual:

`docs/final_presentation/assets/dense_form_resolution_visual.png`

## Text Below Visual

The success case shows why spatial expansion matters: retrieval may find the label, but the answer value often appears nearby rather than on the exact same line.

## Speaker Notes

The specimen example shows the ideal pipeline behaviour. The system found the relevant label and then expanded spatially to include the value below it. The Eastern Airlines example shows why the quality gate is important: weak OCR was rejected, and vision-only succeeded. The date example is the limitation: if OCR confidently misreads handwriting, the model may copy the wrong OCR hint.

---

# Slide 13: Conclusion

**Time:** 0:20

## On-Slide Content

**Conclusion**

- Phi-3.5-Vision is efficient, but dense documents expose resolution and grounding limits.
- Adaptive OCR infusion improves document QA without fine-tuning.
- The model always generates the answer; OCR only provides optional hints.
- Best gains appear on structured, layout-heavy documents.

**Final takeaway:**  
System design can unlock stronger document understanding from small vision-language models.

## Speaker Notes

The main conclusion is that we do not always need to scale the model first. By improving the input pipeline with routing, retrieval, spatial expansion, and gating, we can make a small vision-language model significantly more useful for document QA.

---

# Backup Slide: Prompt Structure

Use this only if you have extra time or if someone asks how OCR was inserted.

## On-Slide Content

```text
[Image]

Answer briefly with only the exact value or phrase from the document.
Do not use full sentences.

OCR hints (may be incomplete; prefer the image if hints lack the answer):
[x1,y1,x2,y2] line 1
[x1,y1,x2,y2] line 2

Question: {question}
```

## Speaker Notes

The prompt is designed to keep the image primary. OCR is explicitly described as incomplete, so the model is encouraged to use it as support rather than blindly trusting it.

---

# Backup Slide: Methodology Details

Use this only if a technical examiner asks for more detail.

## Hybrid Retrieval

```text
final_score = 0.7 * semantic_score + 0.3 * keyword_score
```

- Semantic similarity uses MiniLM embeddings
- OCR lines are embedded with three-line context windows
- Keyword matching catches exact field names
- Minimum retrieval score: 0.20

## Quality Gate

OCR snippet is accepted only if:

- Top retrieval score is at least 0.45
- Snippet adds information beyond the question
- Snippet contains answer-like content

## Spatial Expansion

- Add vertical neighbour below within 45 pixels
- Add value below a detected label in the same column
- Add same-row value to the right with 6 pixel alignment tolerance

---

# How to Fit This Into Slides

Use large, minimal text on the actual slides and keep the explanation in speaker notes.

Recommended visual balance:

- Slides 1 to 4: mostly conceptual, simple diagrams, low text
- Slides 5 to 7: architecture-heavy, use the prepared diagrams
- Slides 8 to 9: pipeline detail — hybrid retrieval, spatial expansion, formatting, and quality gate
- Slide 10: metric explanation with one formula and a tiny example table
- Slide 11: results table plus optional bar chart
- Slide 12: case-study slide with one visual and three compact examples
- Slide 13: short closing slide

Suggested design rule:

- Maximum 4 to 5 bullets per slide
- Highlight only the key numbers: **0.506**, **0.658**, **+0.152**, **+0.202**
- Use captions under diagrams so the audience understands what to look at before you explain it

