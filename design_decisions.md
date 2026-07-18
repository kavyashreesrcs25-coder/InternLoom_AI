# InternLoom AI — Design Decisions

This document explains every major architectural and algorithmic decision
made during the development of the InternLoom AI Resume Shortlisting Engine.

---

## 1. Multi-Column Parsing Strategy

### Problem
Resumes created with Canva, LinkedIn templates, or Word two-column layouts
store text in a fragmented order when extracted naively.
A standard left-to-right read produces garbled output like:
`"Python | Data Science React | Machine Learning"` instead of two clean columns.

### Solution — PyMuPDF Block Sorting
PyMuPDF (`fitz`) exposes text as bounding-box blocks `(x0, y0, x1, y1, text)`.
We sort these blocks by **vertical band first, then horizontal position**:

```python
blocks_sorted = sorted(
    blocks,
    key=lambda b: (round(b[1] / 20) * 20, b[0])  # (band_y, x0)
)
```

Grouping Y coordinates into 20px bands before sorting allows blocks at the
same visual row to sort left-to-right, while still maintaining top-to-bottom
page order. This correctly reconstructs two-column reading order.

### Fallback Chain
If PyMuPDF produces fewer than 50 words, the system automatically tries:
1. `pdfplumber` — better for embedded tables and grid layouts
2. `pdfminer.six` — deep character stream, handles unusual font encodings
3. `EasyOCR` — for fully image-based / scanned PDFs

The strategy that yields the most text is selected.

---

## 2. OCR Fallback Strategy

### Trigger Condition
OCR is triggered when extracted word count < `OCR_WORD_THRESHOLD` (50).
This avoids running the expensive OCR pipeline on clear digital PDFs.

### Pipeline
```
PDF bytes
  → PyMuPDF rasterise page at 200 DPI  → numpy RGB array
  → OpenCV grayscale + CLAHE contrast enhancement
  → Adaptive threshold (binarise)
  → Deskew (minAreaRect angle correction)
  → EasyOCR readtext (English)
  → Sort results by (band_y, x) for reading order
  → Assemble line-grouped text
```

### Preprocessing Rationale
- **200 DPI** — balances OCR accuracy vs. processing speed. 300 DPI gives
  marginal gains for standard A4 resume fonts while tripling memory use.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) — improves
  legibility of low-contrast scans without over-brightening dark backgrounds.
- **Adaptive threshold** — handles uneven lighting across scanned pages far
  better than global Otsu thresholding.
- **Deskew** — corrects up to ±45° page tilt. Skewed text dramatically
  reduces OCR token accuracy.

---

## 3. Skill Extraction Method

### Why full-text scan, not section-only?
Candidates frequently mention skills outside the "Skills" section:
- Projects: *"Built a REST API using FastAPI and PostgreSQL"*
- Experience: *"Worked with TensorFlow and Scikit-learn"*
- Certifications: *"AWS Certified Solutions Architect"*
- Education: *"Machine Learning specialisation — Coursera"*

Restricting skill extraction to the Skills section would miss 30–40% of
actual technical exposure.

### Extraction Tiers
1. **Exact keyword scan** — word-boundary-aware regex against 120+ known skills
2. **Alias scan** — resolves `JS → JavaScript`, `Node → Node.js`, etc.
3. **Bullet-list scan** — detects comma/pipe separated skill lists anywhere in text
4. **Semantic matching at scoring time** — catches near-matches for JD skills
   that weren't in the static taxonomy

### Why a static taxonomy + aliases vs. pure NER?
- NER models (spaCy) are trained on news/general text, not resumes.
  They frequently misclassify tech terms (e.g., "Go" the language vs. verb).
- The static taxonomy + alias table gives deterministic, auditable results
  that recruiters can inspect and extend via `config.py`.
- Semantic similarity (Tier 4) covers the long tail of unseen technologies.

---

## 4. Confidence Calculation

### Three-level system

| Level  | Conditions |
|--------|-----------|
| High   | Clean digital parse + completeness ≥ 50% + ≤ 2 penalties |
| Medium | Partial / OCR parse, OR completeness 30–50%, OR 3 penalties |
| Low    | Failed parse, OR completeness < 30%, OR > 3 penalties |

### Field completeness weighting
Not all fields carry equal importance. The completeness score is
**weighted**, not a simple field count:

| Field         | Weight |
|---------------|--------|
| Skills        | 20     |
| Name          | 15     |
| Email         | 10     |
| College       | 8      |
| Degree        | 8      |
| CGPA/Grade    | 10     |
| Projects      | 10     |
| Graduation Yr | 5      |
| Phone         | 5      |
| Internships   | 5      |
| Certs         | 4      |

A resume missing only "Certifications" still scores 96% completeness.
One missing "Skills" and "Name" scores ≤ 65%.

### Confidence vs. score
Confidence is a **metadata signal**, not a score modifier.
The raw score reflects what was found; confidence tells the recruiter
how much to trust it. Low-confidence candidates are flagged visually
in the dashboard but not artificially scored down, preserving fairness.

---

## 5. Candidate Scoring Logic

### Weight auto-adjustment
When a component has no data (e.g., CGPA not found), its weight is
redistributed proportionally among components that do have data:

```python
# If CGPA (weight=10) is unavailable and all others are available:
# Each other component gains: weight += (10 / 90) × weight
```

This ensures the score always reflects 100% of available information,
rather than penalising candidates whose resumes simply don't mention CGPA.

### CGPA scoring with JD minimum
```
base_score = cgpa / 10.0
if jd_min_cgpa:
    if cgpa < jd_min_cgpa:
        penalty = (min_cgpa - cgpa) / min_cgpa × 0.3
        score = base - penalty
```

The 0.3 penalty cap means even a candidate 1 full CGPA point below the
minimum doesn't score zero — they still contribute their base fraction.

### Project scoring (non-linear)
```
0 projects → 0.0
1 project  → 0.4
2 projects → 0.7
3+         → 1.0
```
Non-linear because a single project demonstrates basic capability,
but 3+ demonstrates consistent practice. A linear 0.33/0.66/1.0
under-rewards candidates with many projects.

### Reason generation
Three reasons are algorithmically generated per candidate:
1. **Skills** — Quantified match fraction with named gaps
2. **Academic** — CGPA + experience count summary
3. **Overall** — Verdict language calibrated to score bracket

This gives recruiters an instant human-readable summary without
requiring them to inspect individual score components.

---

## 6. Shortlisting Thresholds

| Status      | Threshold | Rationale |
|-------------|-----------|-----------|
| Shortlisted | ≥ 70      | Strong fit across most dimensions |
| Reserve     | 50–69     | Partial fit — worth a second look |
| Rejected    | < 50      | Insufficient alignment with JD |

Thresholds are configurable in `config.py`. For highly competitive roles,
recruiters can raise `SHORTLIST_THRESHOLD` to 80.

---

## 7. Modular Architecture Rationale

The project is split into four independent layers:

```
parser/   → Pure extraction, no scoring logic
matcher/  → Pure scoring, no UI logic
ui/       → Pure display, no business logic
utils/    → Shared helpers, no domain logic
```

Benefits:
- Each layer can be unit tested independently
- The scoring engine can be called from CLI without Streamlit
- The parser can be swapped (e.g., replace pdfplumber with a new library)
  without touching the scoring or UI layers
- Config centralisation means all threshold tuning happens in one file
