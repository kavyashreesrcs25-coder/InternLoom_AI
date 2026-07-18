"""
InternLoom AI - Central Configuration
All constants, weights, thresholds, and paths live here.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESUME_DIR = DATA_DIR / "resumes"
JD_DIR = DATA_DIR / "job_descriptions"
OUTPUT_DIR = DATA_DIR / "outputs"
TEMP_RESUME_DIR = DATA_DIR / "temp_resumes"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure required directories exist at runtime
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_RESUME_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# OUTPUT FILE NAMES
# ─────────────────────────────────────────────
SAMPLE_OUTPUT_CSV         = "sample_output.csv"
RANKING_CSV               = "ranking.csv"
PARSE_QUALITY_CSV         = "parse_quality_report.csv"
ANALYTICS_CSV             = "analytics.csv"
CANDIDATE_BEST_MATCH_CSV  = "candidate_best_match.csv"
JOB_WISE_RANKING_CSV      = "job_wise_ranking.csv"
SHORTLISTED_CSV           = "shortlisted.csv"
RESERVE_CSV               = "reserve.csv"
REJECTED_CSV              = "rejected.csv"

# ─────────────────────────────────────────────
# PARSING THRESHOLDS
# ─────────────────────────────────────────────
OCR_WORD_THRESHOLD = 50          # Trigger OCR if fewer than this many words extracted
MIN_TEXT_LENGTH = 20             # Minimum characters for a valid parse
MAX_PDF_PAGES = 10               # Maximum pages to process per resume

# ─────────────────────────────────────────────
# SCORING WEIGHTS (must sum to 100)
# ─────────────────────────────────────────────
SCORING_WEIGHTS = {
    "required_skills": 40,
    "preferred_skills": 20,
    "projects": 15,
    "cgpa": 10,
    "internship": 10,
    "certifications": 5,
}

# ─────────────────────────────────────────────
# SHORTLISTING THRESHOLDS
# ─────────────────────────────────────────────
SHORTLIST_THRESHOLD = 70         # Score >= this → Shortlisted
RESERVE_THRESHOLD = 50           # Score >= this → Reserve
# Below RESERVE_THRESHOLD → Rejected

# ─────────────────────────────────────────────
# CONFIDENCE LEVELS
# ─────────────────────────────────────────────
CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"

PARSE_STATUS_CLEAN = "Clean"
PARSE_STATUS_PARTIAL = "Partial"
PARSE_STATUS_OCR = "OCR"
PARSE_STATUS_FAILED = "Failed"

CONFIDENCE_MAP = {
    PARSE_STATUS_CLEAN: CONFIDENCE_HIGH,
    PARSE_STATUS_PARTIAL: CONFIDENCE_MEDIUM,
    PARSE_STATUS_OCR: CONFIDENCE_MEDIUM,
    PARSE_STATUS_FAILED: CONFIDENCE_LOW,
}

# ─────────────────────────────────────────────
# GRADE NORMALIZATION
# ─────────────────────────────────────────────
CGPA_10_MAX = 10.0
GPA_4_MAX = 4.0
PERCENTAGE_DIVISOR = 9.5         # Percentage ÷ 9.5 → 10-point CGPA equivalent
GPA_4_MULTIPLIER = 2.5           # 4-point GPA × 2.5 → 10-point equivalent

# ─────────────────────────────────────────────
# SEMANTIC SIMILARITY
# ─────────────────────────────────────────────
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_SIMILARITY_THRESHOLD = 0.65    # Cosine similarity threshold
FUZZY_MATCH_THRESHOLD = 85              # RapidFuzz token_sort_ratio threshold

# ─────────────────────────────────────────────
# SKILL TAXONOMY — Core known skills
# ─────────────────────────────────────────────
KNOWN_SKILLS = [
    # Programming Languages
    "Python", "Java", "C", "C++", "C#", "Go", "Rust", "Kotlin", "Swift",
    "JavaScript", "TypeScript", "PHP", "Ruby", "Scala", "R", "MATLAB",
    "Bash", "Shell", "PowerShell", "Perl", "Lua",
    # Web Frontend
    "React", "ReactJS", "Next.js", "Vue.js", "Angular", "Svelte",
    "HTML", "CSS", "SASS", "SCSS", "Tailwind CSS", "Bootstrap",
    "Redux", "Zustand", "Webpack", "Vite",
    # Web Backend
    "Node.js", "Express", "Express.js", "FastAPI", "Flask", "Django",
    "Spring Boot", "Laravel", "Rails", "ASP.NET", "NestJS",
    # Databases
    "MongoDB", "MySQL", "PostgreSQL", "SQLite", "Redis", "Cassandra",
    "DynamoDB", "Elasticsearch", "Neo4j", "Oracle", "MSSQL",
    "Firebase", "Supabase", "PlanetScale",
    # APIs & Protocols
    "REST API", "GraphQL", "gRPC", "WebSocket", "OAuth", "JWT",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes",
    "Terraform", "Ansible", "Jenkins", "GitHub Actions", "CI/CD",
    "Nginx", "Apache", "Linux", "Ubuntu",
    # Version Control
    "Git", "GitHub", "GitLab", "Bitbucket",
    # Data Science & ML
    "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost",
    "Machine Learning", "Deep Learning", "Artificial Intelligence",
    "Data Science", "NLP", "Computer Vision", "Reinforcement Learning",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
    "Jupyter", "Anaconda",
    # AI/LLM
    "LLM", "Prompt Engineering", "LangChain", "LlamaIndex",
    "Vector Database", "RAG", "OpenAI", "Hugging Face",
    "Transformers", "BERT", "GPT",
    # Mobile
    "Android", "iOS", "React Native", "Flutter", "Dart",
    # Testing
    "Pytest", "Jest", "Selenium", "Cypress", "Postman", "JUnit",
    # Other Tools
    "Kafka", "RabbitMQ", "Celery", "Airflow", "Spark", "Hadoop",
    "Power BI", "Tableau", "Excel", "Figma", "Jira", "Confluence",
    # Security
    "Cybersecurity", "Penetration Testing", "Cryptography",
    "OWASP", "SSL/TLS",
    # Blockchain
    "Blockchain", "Solidity", "Web3", "Ethereum", "Smart Contracts",
]

# ─────────────────────────────────────────────
# TECHNOLOGY ALIASES / SYNONYMS
# ─────────────────────────────────────────────
SKILL_ALIASES: dict[str, list[str]] = {
    "React": ["ReactJS", "React.js", "React JS"],
    "Node.js": ["NodeJS", "Node", "Node JS"],
    "JavaScript": ["JS", "ECMAScript", "ES6", "ES2015"],
    "TypeScript": ["TS"],
    "PostgreSQL": ["Postgres", "SQL"],
    "MySQL": ["SQL"],
    "MongoDB": ["Mongo"],
    "REST API": ["REST", "RESTful", "RESTful API"],
    "Machine Learning": ["ML"],
    "Deep Learning": ["DL"],
    "Artificial Intelligence": ["AI"],
    "Natural Language Processing": ["NLP"],
    "Computer Vision": ["CV"],
    "C++": ["CPP", "C Plus Plus"],
    "C#": ["CSharp", "C Sharp"],
    "Python": ["Py"],
    "TensorFlow": ["TF"],
    "PyTorch": ["Torch"],
    "Scikit-learn": ["sklearn", "Scikit Learn"],
    "GitHub Actions": ["GH Actions"],
    "Google Cloud": ["GCP", "Google Cloud Platform"],
    "Amazon Web Services": ["AWS"],
    "Microsoft Azure": ["Azure"],
    "Next.js": ["NextJS", "Next JS"],
    "Vue.js": ["VueJS", "Vue"],
    "Express.js": ["Express", "ExpressJS"],
    "React Native": ["RN"],
    "Flutter": ["Dart/Flutter"],
    "Large Language Model": ["LLM"],
    "Retrieval Augmented Generation": ["RAG"],
}

# Build reverse alias map: alias → canonical
ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in SKILL_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical

# ─────────────────────────────────────────────
# SECTION HEADER KEYWORDS (for resume parsing)
# ─────────────────────────────────────────────
SECTION_HEADERS = {
    "skills": [
        "skills", "technical skills", "core competencies",
        "technologies", "tech stack", "tools", "expertise",
        "technical expertise", "programming languages",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment", "work history", "career history",
        "professional background",
    ],
    "internship": [
        "internship", "internships", "intern", "industrial training",
        "summer internship", "training",
    ],
    "education": [
        "education", "academic background", "educational qualification",
        "academics", "qualifications", "degrees",
    ],
    "projects": [
        "projects", "personal projects", "academic projects",
        "project work", "side projects", "notable projects",
        "key projects",
    ],
    "certifications": [
        "certifications", "certificates", "courses",
        "online courses", "achievements", "awards",
        "professional certifications", "licenses",
    ],
    "summary": [
        "summary", "objective", "profile", "about me",
        "professional summary", "career objective",
        "personal statement",
    ],
}

# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────
EMAIL_PATTERN = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = (
    r"(?:\+91[\-\s]?)?(?:\(?[0-9]{3}\)?[\-\s]?[0-9]{3}[\-\s]?[0-9]{4}"
    r"|[6-9]\d{9})"
)
GITHUB_PATTERN = r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+"
LINKEDIN_PATTERN = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+"
PORTFOLIO_PATTERN = (
    r"(?:https?://)?(?:www\.)?(?!github|linkedin)"
    r"[\w\-]+\.(?:com|io|dev|me|co|net|org)/[\w\-/]*"
)
CGPA_PATTERN = (
    r"(?:cgpa|gpa|cpi|spi|grade|score)[^\d]*(\d+\.?\d*)"
    r"\s*(?:/\s*(\d+\.?\d*))?"
)
PERCENTAGE_PATTERN = r"(\d{2,3}(?:\.\d+)?)\s*%"
GRAD_YEAR_PATTERN = r"(?:20[1-3]\d)"

# ─────────────────────────────────────────────
# JD EXTRACTION KEYWORDS
# ─────────────────────────────────────────────
JD_REQUIRED_KEYWORDS = [
    "required", "must have", "mandatory", "essential",
    "minimum requirements", "qualifications",
]
JD_PREFERRED_KEYWORDS = [
    "preferred", "nice to have", "good to have",
    "plus", "bonus", "desired", "advantageous",
]
JD_RESPONSIBILITY_KEYWORDS = [
    "responsibilities", "duties", "you will", "role involves",
    "key responsibilities", "what you'll do",
]

# ─────────────────────────────────────────────
# APP UI
# ─────────────────────────────────────────────
APP_TITLE    = "InternLoom AI"
APP_SUBTITLE = "Intelligent Multi-JD Resume Shortlisting Engine"
APP_ICON     = "🎯"
APP_VERSION  = "2.0.0"

PAGE_NAMES = {
    "home":             "🏠 Home",
    "upload_resumes":   "📂 Upload Resumes",
    "upload_jd":        "📄 Upload Job Descriptions",
    "analysis":         "🔬 Analyze",
    "best_match":       "🎯 Candidate Best Match",
    "job_ranking":      "🏆 Job-wise Ranking",
    "analytics":        "📈 Analytics",
    "reports":          "📑 Reports & Export",
    "settings":         "⚙️ Settings",
    "about":            "ℹ️ About",
}

# Plotly color palette
CHART_COLORS = [
    "#6C63FF", "#FF6584", "#43BCCD", "#F7B731",
    "#45AAB8", "#26de81", "#fd9644", "#a55eea",
]

PRIMARY_COLOR = "#6C63FF"
SUCCESS_COLOR = "#26de81"
WARNING_COLOR = "#F7B731"
DANGER_COLOR = "#FF6584"
INFO_COLOR = "#43BCCD"
