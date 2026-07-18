# 🎯 InternLoom AI — Intelligent Resume Shortlisting Engine

> A production-ready, AI-powered resume screening system that parses PDFs,
> extracts structured candidate data, scores applicants against job descriptions,
> and generates recruiter-ready ranked reports.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| Multi-strategy PDF parsing | PyMuPDF → pdfplumber → pdfminer.six → EasyOCR (automatic fallback) |
| Layout handling | Single-column, two-column, Canva templates, tables, scanned PDFs |
| Field extraction | Name, email, phone, college, degree, branch, grad year, CGPA, skills, projects, internships, certifications, GitHub, LinkedIn |
| Grade normalisation | 10-pt CGPA · 4-pt GPA (×2.5) · Percentage (÷9.5) · Unknown scale (inferred) |
| Skill matching | Exact · Alias/synonym · RapidFuzz token-sort · Semantic (Sentence Transformers) |
| Scoring | Weighted 0–100 score with auto weight adjustment for missing fields |
| Explainability | 3 human-readable reasons per candidate |
| Confidence | High / Medium / Low based on parse quality + field completeness |
| Shortlisting | Shortlisted ≥ 70 · Reserve ≥ 50 · Rejected < 50 |
| Reports | `sample_output.csv`, `ranking.csv`, `parse_quality_report.csv`, `analytics.csv` |
| Dashboard | 8-page Streamlit dark-mode UI with Plotly charts |

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/your-org/InternLoom_AI.git
cd InternLoom_AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
InternLoom_AI/
├── app.py                      # Entry point — streamlit run app.py
├── config.py                   # All constants, weights, patterns
├── requirements.txt
├── README.md
├── design_decisions.md
├── ai_usage_log.md
│
├── data/
│   ├── resumes/                # Uploaded resumes stored here
│   ├── job_descriptions/       # Uploaded JDs stored here
│   └── outputs/                # Generated CSV reports
│
├── parser/
│   ├── pdf_parser.py           # Multi-strategy PDF text extraction
│   ├── ocr.py                  # EasyOCR + OpenCV preprocessing
│   ├── extractor.py            # Structured field extraction
│   └── normalizer.py           # Grade normalisation + JD parser
│
├── matcher/
│   ├── skill_matcher.py        # 3-tier skill matching engine
│   ├── scorer.py               # Weighted scoring engine
│   ├── confidence.py           # Parse confidence calculator
│   └── ranking.py              # Full pipeline orchestrator + reports
│
├── ui/
│   └── dashboard.py            # Streamlit 8-page dashboard
│
└── utils/
    ├── logger.py               # Structured logging
    └── helper.py               # Shared utility functions
```

---

## 🔧 Configuration

All tunable parameters live in `config.py`:

```python
# Scoring weights (must sum to 100)
SCORING_WEIGHTS = {
    "required_skills": 40,
    "preferred_skills": 20,
    "projects":         15,
    "cgpa":             10,
    "internship":       10,
    "certifications":    5,
}

# Shortlisting thresholds
SHORTLIST_THRESHOLD = 70   # Score >= 70 → Shortlisted
RESERVE_THRESHOLD   = 50   # Score >= 50 → Reserve

# OCR trigger
OCR_WORD_THRESHOLD  = 50   # Trigger OCR if fewer than 50 words extracted
```

---

## 📊 Scoring Logic

Each candidate receives a score out of **100** computed as:

```
Score = Σ (component_raw_score × component_weight)
```

If a component lacks data (e.g., CGPA not found), its weight is
**redistributed proportionally** among available components so the
total always sums to 100.

### Skill Matching Tiers

1. **Exact** — case-insensitive literal match (score: 1.0)
2. **Alias** — technology synonyms (React = ReactJS, JS = JavaScript …) (score: 0.95)
3. **Fuzzy** — RapidFuzz token-sort ratio ≥ 85 (score: proportional)
4. **Semantic** — Sentence Transformers cosine similarity ≥ 0.65 (score: similarity)

---

## 📑 Output Files

All CSVs are saved to `data/outputs/` and available for download in the UI.

| File | Contents |
|------|----------|
| `sample_output.csv` | All extracted fields + scores for every candidate |
| `ranking.csv` | Ranked shortlist with score, status, matched skills, reasons |
| `parse_quality_report.csv` | Parse status, word count, field completeness per resume |
| `analytics.csv` | Skill frequency, score distribution, status counts |

---

## 🧠 Model Downloads (auto on first use)

| Model | Size | Purpose |
|-------|------|---------|
| `all-MiniLM-L6-v2` | ~90 MB | Semantic skill similarity |
| `en_core_web_sm` | ~12 MB | spaCy NER for name extraction |
| EasyOCR weights | ~100 MB | OCR for scanned PDFs |

All models are cached locally after the first download.

---

## ⚙️ Optional: OpenAI / LangChain

Uncomment the LangChain/OpenAI lines in `requirements.txt` and set:

```bash
# .env
OPENAI_API_KEY=sk-...
```

This enables LLM-enhanced JD parsing and richer score explanations.

---

## 📜 License

MIT License — free for personal and commercial use.
