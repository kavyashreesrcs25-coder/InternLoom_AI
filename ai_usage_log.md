# InternLoom AI — AI Usage Log

This document records all AI model usage within the system:
what models are used, why, where, and what data they process.

---

## Models in Use

### 1. `all-MiniLM-L6-v2` — Sentence Transformers

| Attribute | Detail |
|-----------|--------|
| **Library** | `sentence-transformers` |
| **Source** | Hugging Face Hub (auto-downloaded ~90 MB) |
| **Purpose** | Semantic skill similarity matching |
| **Used in** | `matcher/skill_matcher.py` → `_semantic_match()` |
| **Input** | Skill name strings (e.g., "React", "Node.js") |
| **Output** | 384-dimensional embedding vectors |
| **Trigger** | Only when Tier 1 (exact) and Tier 2 (alias/fuzzy) matching fail |
| **Privacy** | All inference is **local** — no data sent externally |
| **Caching** | Embeddings cached in-memory per session to avoid recomputation |

**Why this model?**
`all-MiniLM-L6-v2` is the best balance of speed and quality for
short-text semantic similarity. At 80 MB it fits comfortably in RAM,
runs in ~5ms per embedding on CPU, and achieves near-SOTA performance
on STS (Semantic Textual Similarity) benchmarks for skill-length strings.

---

### 2. `en_core_web_sm` — spaCy NER

| Attribute | Detail |
|-----------|--------|
| **Library** | `spacy` |
| **Source** | spaCy model hub (manual download: `python -m spacy download en_core_web_sm`) |
| **Purpose** | Named Entity Recognition for candidate name extraction |
| **Used in** | `parser/extractor.py` → `_extract_name_spacy()` |
| **Input** | First 500 characters of resume text |
| **Output** | PERSON entity spans |
| **Fallback** | Heuristic capitalisation scan if spaCy unavailable or no PERSON found |
| **Privacy** | All inference is **local** |

**Why only name extraction?**
spaCy's `en_core_web_sm` is trained on general English text (news, web).
It performs well on person names but poorly on technology terms.
Skill extraction is handled by the deterministic taxonomy + alias engine,
which is more reliable and auditable for technical vocabulary.

---

### 3. EasyOCR (CRAFT + CRNN)

| Attribute | Detail |
|-----------|--------|
| **Library** | `easyocr` |
| **Source** | Auto-downloaded on first OCR call (~100 MB) |
| **Purpose** | Text extraction from scanned / image PDFs |
| **Used in** | `parser/ocr.py` → `OCRExtractor` |
| **Input** | Numpy image arrays (rendered PDF pages at 200 DPI) |
| **Output** | List of (bounding_box, text, confidence) tuples |
| **Trigger** | Only when digital extraction yields < 50 words |
| **Privacy** | All inference is **local** |
| **GPU** | Disabled by default (`gpu=False`). Set `gpu=True` for faster OCR. |

**Architecture:**
- **CRAFT** (Character Region Awareness for Text Detection) — detects
  text regions in the image
- **CRNN** (Convolutional Recurrent Neural Network) — recognises
  characters within each detected region

---

## Optional AI Components (not active by default)

### 4. OpenAI GPT (optional)

| Attribute | Detail |
|-----------|--------|
| **Library** | `openai`, `langchain-openai` |
| **Purpose** | LLM-enhanced JD parsing, richer score explanations |
| **Status** | Commented out in `requirements.txt` |
| **Activation** | Uncomment langchain/openai deps + set `OPENAI_API_KEY` in `.env` |
| **Privacy** | ⚠️ Resume text would be sent to OpenAI API — review your data policy before enabling |
| **Cost** | GPT-4o-mini: ~$0.0002 per resume at typical resume length |

---

## Data Flow Summary

```
PDF bytes (local)
    ↓ PyMuPDF / pdfplumber / pdfminer (local)
    ↓ EasyOCR if needed (local)
    ↓ spaCy NER — first 500 chars (local)
    ↓ Regex extraction (local)
    ↓ Sentence Transformers embeddings (local)
    ↓ RapidFuzz fuzzy matching (local)
    ↓ Score + Rank
    ↓ Streamlit UI display (local)
    ↓ CSV export (local)
```

**All processing is entirely local. No resume data, candidate PII,
or JD content is transmitted to any external service unless
the optional OpenAI integration is explicitly enabled.**

---

## Responsible AI Notes

- **Bias awareness:** The scoring system uses objective signals
  (skills, CGPA, project count) only. It does not use name, gender,
  nationality, photo, or any protected attribute.
- **Transparency:** Every score comes with 3 human-readable reasons
  and a full per-component breakdown. Recruiters can inspect and
  override any recommendation.
- **Confidence flagging:** Low-confidence candidates (poor parse quality)
  are explicitly flagged so recruiters know to review them manually
  rather than relying solely on the automated score.
- **No automated rejection:** The system classifies candidates as
  "Shortlisted / Reserve / Rejected" but all final decisions
  must be made by a human recruiter.
