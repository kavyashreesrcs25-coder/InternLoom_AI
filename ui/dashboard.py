"""
InternLoom AI v2 - Streamlit Dashboard
10 pages: Home · Upload Resumes · Upload JDs · Analyze ·
          Candidate Best Match · Job-wise Ranking ·
          Analytics · Reports · Settings · About
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON, APP_VERSION, PAGE_NAMES,
    PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR, INFO_COLOR,
    SHORTLIST_THRESHOLD, RESERVE_THRESHOLD, SCORING_WEIGHTS, CHART_COLORS,
)
from utils.helper import df_to_csv_bytes, score_color, status_badge, confidence_badge

_CSS = """<style>
[data-testid="stAppViewContainer"]{background:#0f1117;color:#e0e0e0;}
[data-testid="stSidebar"]{background:#161b27;border-right:1px solid #2a2f3e;}
.kpi-card{background:linear-gradient(135deg,#1e2235,#252b3e);border-radius:12px;
  padding:18px 12px;text-align:center;border:1px solid #2e3550;margin-bottom:8px;
  box-shadow:0 4px 14px rgba(0,0,0,.3);}
.kpi-value{font-size:2rem;font-weight:700;margin-bottom:3px;}
.kpi-label{font-size:.78rem;color:#8892b0;text-transform:uppercase;letter-spacing:1px;}
.section-header{font-size:1.35rem;font-weight:700;color:#6C63FF;
  border-bottom:2px solid #2a2f3e;padding-bottom:5px;margin-bottom:12px;}
.skill-chip{display:inline-block;background:#252b3e;border:1px solid #3a4060;
  border-radius:14px;padding:2px 9px;font-size:.76rem;margin:2px;color:#aab0cc;}
.skill-chip.matched{background:#1a3a2a;border-color:#26de81;color:#26de81;}
.skill-chip.missing{background:#3a1a25;border-color:#FF6584;color:#FF6584;}
.sidebar-logo{text-align:center;padding:14px 0 6px;font-size:1.5rem;font-weight:800;
  background:linear-gradient(90deg,#6C63FF,#43BCCD);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
</style>"""

_DEFAULTS = {
    "uploaded_pdfs":       [],
    "gdrive_loaded_names": set(),
    "gdrive_last_result":  None,
    "jd_list":             [],
    "multi_results":       [],
    "multi_reports":       {},
    "jobwise_rankings":    {},
    "allow_multi_match":   False,
    "use_semantic":        True,
    "current_page":        "home",
}

# ── Shared helpers ─────────────────────────────────────────────────────────

def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)

def kpi_card(label, value, color=PRIMARY_COLOR):
    return (f'<div class="kpi-card"><div class="kpi-value" style="color:{color};">'
            f'{value}</div><div class="kpi-label">{label}</div></div>')

def section_header(t):
    st.markdown(f'<div class="section-header">{t}</div>', unsafe_allow_html=True)

def alert_box(msg, kind="info"):
    c, i = {"info":(INFO_COLOR,"ℹ️"),"success":(SUCCESS_COLOR,"✅"),
             "warning":(WARNING_COLOR,"⚠️"),"error":(DANGER_COLOR,"❌")}.get(kind,(INFO_COLOR,"ℹ️"))
    st.markdown(
        f'<div style="background:#1a1f2e;border-left:4px solid {c};'
        f'padding:11px 15px;border-radius:6px;margin:7px 0;">{i} {msg}</div>',
        unsafe_allow_html=True)

def skill_chips(skills, cls=""):
    if not skills:
        return '<span style="color:#555;">—</span>'
    return '<div style="line-height:2.1;">' + "".join(
        f'<span class="skill-chip {cls}">{s}</span>' for s in skills) + '</div>'

def progress_row(label, value, max_val=100.0):
    pct = min(value / max_val, 1.0) if max_val else 0.0
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:.8rem;'
        f'color:#8892b0;margin-bottom:2px;"><span>{label}</span><span>{value:.1f}</span></div>',
        unsafe_allow_html=True)
    st.progress(pct)

def _plotly_dark(fig):
    fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#141824",
                      font_color="#c0c0d0", margin=dict(l=20,r=20,t=40,b=20),
                      legend=dict(bgcolor="#1a1f2e", bordercolor="#2a2f3e"))

def _dl_btn(label, df, fname):
    if df is not None and not df.empty:
        st.download_button(label, df_to_csv_bytes(df), fname, "text/csv",
                           use_container_width=True)

def _file_row(fname, fbytes, icon="💾"):
    kb = len(fbytes) / 1024
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'background:#1a1f2e;padding:9px 13px;border-radius:8px;margin-bottom:5px;">'
        f'<span>{icon} 📄 <strong>{fname}</strong></span>'
        f'<span style="color:#8892b0;font-size:.83rem;">{kb:.1f} KB</span></div>',
        unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🎯 InternLoom AI</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center;color:#555;font-size:.72rem;">v{APP_VERSION}</p>',
                    unsafe_allow_html=True)
        st.divider()
        labels = list(PAGE_NAMES.values())
        keys   = list(PAGE_NAMES.keys())
        if "current_page" not in st.session_state:
            st.session_state.current_page = "home"
        try:
            idx = keys.index(st.session_state.current_page)
        except ValueError:
            idx = 0
        sel = st.radio("Nav", labels, index=idx, label_visibility="collapsed")
        key = keys[labels.index(sel)]
        st.session_state.current_page = key
        st.divider()
        n_res   = len(st.session_state.get("uploaded_pdfs", []))
        n_jd    = len(st.session_state.get("jd_list", []))
        n_done  = len(st.session_state.get("multi_results", []))
        gd      = st.session_state.get("gdrive_last_result")
        st.markdown("**Pipeline Status**")
        st.markdown(f"{'✅' if n_res  else '⬜'} {n_res} resume(s) loaded")
        st.markdown(f"{'☁️' if gd and gd.success else '⬜'} Drive import")
        st.markdown(f"{'✅' if n_jd   else '⬜'} {n_jd} JD(s) loaded")
        st.markdown(f"{'✅' if n_done else '⬜'} Analysis {'done' if n_done else 'pending'}")
        st.divider()
        st.caption("InternLoom AI © 2024")
    return key

# ── Page 1: Home ──────────────────────────────────────────────────────────

def page_home():
    st.markdown(
        f'<div style="text-align:center;padding:36px 0 16px;">'
        f'<div style="font-size:3.2rem;">🎯</div>'
        f'<h1 style="font-size:2.6rem;font-weight:800;background:linear-gradient(90deg,#6C63FF,#43BCCD);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{APP_TITLE}</h1>'
        f'<p style="font-size:1.05rem;color:#8892b0;margin-top:-8px;">{APP_SUBTITLE}</p></div>',
        unsafe_allow_html=True)

    results = st.session_state.get("multi_results", [])
    jds     = st.session_state.get("jd_list", [])
    if results:
        from matcher.multi_jd import MultiJDRankingEngine
        s = MultiJDRankingEngine.get_summary_stats(results, jds)
        cols = st.columns(6)
        for col, (lbl, val, clr) in zip(cols, [
            ("Total",       s["total"],              PRIMARY_COLOR),
            ("Shortlisted", s["shortlisted"],        SUCCESS_COLOR),
            ("Reserve",     s["reserve"],            WARNING_COLOR),
            ("Rejected",    s["rejected"],           DANGER_COLOR),
            ("Avg Score",   f"{s['avg_score']:.1f}", INFO_COLOR),
            ("JDs",         s["jd_count"],           "#a55eea"),
        ]):
            col.markdown(kpi_card(lbl, val, clr), unsafe_allow_html=True)
        st.divider()

    section_header("🚀 Quick Start")
    for num, title, desc in [
        ("1","📂 Upload Resumes",        "Drag & drop PDFs or paste a Google Drive folder URL."),
        ("2","📄 Upload Job Descriptions","Upload 1–20+ JDs in PDF, DOCX, or TXT format."),
        ("3","🔬 Analyze",               "Every resume is automatically scored against every JD."),
        ("4","🎯 Candidate Best Match",  "See each candidate's best role, score, and reasons."),
        ("5","🏆 Job-wise Ranking",       "Filter & sort ranked lists per job role."),
        ("6","📑 Reports",               "Download 7 CSV files for recruiter hand-off."),
    ]:
        st.markdown(
            f'<div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:9px;">'
            f'<div style="background:#6C63FF;border-radius:50%;width:26px;height:26px;'
            f'display:flex;align-items:center;justify-content:center;font-weight:700;'
            f'font-size:.82rem;flex-shrink:0;">{num}</div>'
            f'<div><strong style="color:#cdd;">{title}</strong> — {desc}</div></div>',
            unsafe_allow_html=True)

    st.divider()
    section_header("✨ What's New in v2")
    c1, c2, c3 = st.columns(3)
    for col, ico, title, body in [
        (c1, "🔧", "Fixed Drive Import",
         "Downloads by File ID via gdown. Verifies %PDF magic bytes. Retries once on failure."),
        (c2, "📄", "Multi-JD Matching",
         "Upload 20+ JDs. Every resume is scored against every role automatically."),
        (c3, "🤖", "AI Role Recommendation",
         "Candidates with no match get top-3 role suggestions based on their skills."),
    ]:
        with col:
            st.markdown(
                f'<div style="background:#1a1f2e;border-radius:10px;padding:15px;border:1px solid #2a2f3e;">'
                f'<div style="font-size:1.3rem;">{ico}</div>'
                f'<strong style="color:#cdd;">{title}</strong>'
                f'<p style="color:#8892b0;font-size:.85rem;margin-top:5px;">{body}</p></div>',
                unsafe_allow_html=True)

# ── Page 2: Upload Resumes ────────────────────────────────────────────────

def _tab_manual_upload():
    up = st.file_uploader("Drop PDF resumes here", type=["pdf"],
                          accept_multiple_files=True, key="resume_uploader")
    if up:
        new_files   = [(f.name, f.read()) for f in up]
        drive_names = st.session_state.get("gdrive_loaded_names", set())
        existing    = [(f, b) for f, b in st.session_state.get("uploaded_pdfs", [])
                       if f in drive_names]
        seen: set   = {f for f, _ in existing}
        merged      = existing + [(f, b) for f, b in new_files if f not in seen]
        st.session_state["uploaded_pdfs"] = merged
        alert_box(f"✅ **{len(new_files)}** file(s) loaded. Total: **{len(merged)}**.", "success")
    else:
        st.markdown(
            '<div style="text-align:center;padding:44px;background:#1a1f2e;'
            'border-radius:12px;border:2px dashed #2a2f3e;">'
            '<div style="font-size:2.5rem;">📂</div>'
            '<p style="color:#556;margin-top:8px;font-size:.88rem;">'
            'Drag PDF files here or click Browse.</p></div>', unsafe_allow_html=True)
        existing_n = len(st.session_state.get("uploaded_pdfs", []))
        if existing_n:
            alert_box(f"{existing_n} resume(s) already loaded.", "info")


def _tab_gdrive():
    from utils.gdrive import extract_folder_id, import_from_drive

    st.markdown("Paste a **public** Google Drive folder URL. All PDFs are downloaded automatically.")
    with st.expander("ℹ️ How to share a Drive folder publicly"):
        st.markdown("""
1. Right-click the folder → **Share**.
2. Set *General access* → **Anyone with the link → Viewer**.
3. Copy the link and paste below.

```
https://drive.google.com/drive/folders/<ID>
https://drive.google.com/drive/folders/<ID>?usp=sharing
```""")

    url = st.text_input("Google Drive Folder URL",
                        placeholder="https://drive.google.com/drive/folders/...",
                        key="gdrive_url")
    url_ok = False
    if url.strip():
        fid = extract_folder_id(url.strip())
        if fid:
            st.markdown(f'<div style="color:#26de81;font-size:.82rem;">✅ Folder ID: <code>{fid}</code></div>',
                        unsafe_allow_html=True)
            url_ok = True
        else:
            alert_box("Invalid URL — cannot extract folder ID.", "warning")

    o1, o2 = st.columns(2)
    with o1: auto_run = st.checkbox("Auto-run analysis after download", True, key="gd_auto")
    with o2: use_sem  = st.checkbox("Semantic Matching", True, key="gd_sem")

    if not st.session_state.get("jd_list"):
        alert_box("Load at least one JD in **📄 Upload Job Descriptions** before auto-run.", "warning")

    st.markdown("")
    if not st.button("☁️ Import from Google Drive", type="primary",
                     use_container_width=True, disabled=not url_ok, key="gd_btn"):
        prev = st.session_state.get("gdrive_last_result")
        if prev and prev.success:
            alert_box(f"Last import: **{len(prev.downloaded)}** PDFs downloaded.", "info")
        return

    st.divider()
    st.markdown("#### 📥 Downloading…")
    stat_txt = st.empty()
    dl_bar   = st.progress(0.0)
    live_ctr = st.empty()
    ctr      = {"ok": 0, "failed": 0, "skipped": 0}

    def on_dl(cur, tot, fname, status):
        if status in ctr:
            ctr[status] += 1
        if status == "listing":
            stat_txt.markdown("🔍 Listing files in folder…")
            return
        emojis = {"ok": "✅", "failed": "❌", "skipped": "⏭️"}
        stat_txt.markdown(f"{emojis.get(status,'•')} `{fname}` ({cur}/{tot})")
        dl_bar.progress(cur / max(tot, 1))
        live_ctr.markdown(
            f"**✅ {ctr['ok']}** downloaded &nbsp;|&nbsp; "
            f"**❌ {ctr['failed']}** failed &nbsp;|&nbsp; "
            f"**⏭️ {ctr['skipped']}** skipped")

    with st.spinner("Connecting to Google Drive…"):
        result = import_from_drive(url.strip(), on_dl)

    dl_bar.progress(1.0)
    stat_txt.empty()
    st.session_state["gdrive_last_result"] = result

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(kpi_card("Found",      result.total_pdfs,      PRIMARY_COLOR), unsafe_allow_html=True)
    with k2: st.markdown(kpi_card("Downloaded", len(result.downloaded), SUCCESS_COLOR), unsafe_allow_html=True)
    with k3: st.markdown(kpi_card("Failed",     len(result.failed),     DANGER_COLOR),  unsafe_allow_html=True)
    with k4: st.markdown(kpi_card("Skipped",    len(result.skipped),    WARNING_COLOR), unsafe_allow_html=True)

    if result.error:
        alert_box(result.error, "error")
        return
    if result.failed:
        with st.expander(f"⚠️ {len(result.failed)} failure(s)"):
            for fn, err in result.failed:
                st.markdown(f"- **{fn}**: {err}")
    if not result.downloaded:
        alert_box("No PDFs downloaded. Cannot continue.", "error")
        return

    dn = st.session_state.get("gdrive_loaded_names", set())
    dn.update(f for f, _ in result.downloaded)
    st.session_state["gdrive_loaded_names"] = dn
    existing   = st.session_state.get("uploaded_pdfs", [])
    ex_names   = {f for f, _ in existing}
    merged     = existing + [(f, b) for f, b in result.downloaded if f not in ex_names]
    st.session_state["uploaded_pdfs"] = merged
    alert_box(f"✅ **{len(result.downloaded)}** PDFs imported. Total: **{len(merged)}**.", "success")

    jds = st.session_state.get("jd_list", [])
    if auto_run and jds:
        _run_analysis(merged, jds, use_sem)


def page_upload_resumes():
    section_header("📂 Upload Resumes")
    st.markdown("Choose how to load resumes — both methods merge automatically.")
    st.markdown(
        '<div style="display:flex;gap:14px;margin:14px 0;">'
        '<div style="flex:1;background:linear-gradient(135deg,#1e2235,#252b3e);border-radius:12px;'
        'padding:18px;border:1px solid #2e3550;text-align:center;">'
        '<div style="font-size:2.2rem;">💾</div>'
        '<div style="font-weight:700;color:#cdd;margin-top:6px;">Manual Upload</div>'
        '<div style="color:#8892b0;font-size:.8rem;margin-top:3px;">Drag & drop PDFs</div></div>'
        '<div style="flex:1;background:linear-gradient(135deg,#1e2235,#252b3e);border-radius:12px;'
        'padding:18px;border:1px solid #2e3550;text-align:center;">'
        '<div style="font-size:2.2rem;">☁️</div>'
        '<div style="font-weight:700;color:#cdd;margin-top:6px;">Google Drive Import</div>'
        '<div style="color:#8892b0;font-size:.8rem;margin-top:3px;">Paste public folder URL</div></div></div>',
        unsafe_allow_html=True)

    t1, t2 = st.tabs(["💾  Manual Upload", "☁️  Google Drive Import"])
    with t1:
        st.markdown("")
        _tab_manual_upload()
    with t2:
        st.markdown("")
        _tab_gdrive()

    pdfs = st.session_state.get("uploaded_pdfs", [])
    if pdfs:
        st.divider()
        section_header(f"📋 Loaded Resumes ({len(pdfs)})")
        dn = st.session_state.get("gdrive_loaded_names", set())
        for fname, fbytes in pdfs:
            _file_row(fname, fbytes, "☁️" if fname in dn else "💾")
        st.markdown("")
        alert_box("Next: go to **📄 Upload Job Descriptions**.", "info")
        if st.button("🗑️ Clear all resumes", key="clear_res"):
            st.session_state.update({"uploaded_pdfs": [], "gdrive_loaded_names": set(),
                                     "gdrive_last_result": None})
            st.rerun()

# ── Page 3: Upload Multiple JDs ───────────────────────────────────────────

def page_upload_jd():
    from parser.normalizer import parse_jd_file, parse_job_description

    section_header("📄 Upload Job Descriptions")
    st.markdown("Upload **multiple** JDs (PDF, DOCX, TXT) or paste text. Each is parsed independently.")

    tab_files, tab_paste = st.tabs(["📁 Upload Files", "✏️ Paste JD Text"])

    with tab_files:
        uploaded = st.file_uploader("Upload JD files (PDF / DOCX / TXT)",
                                    type=["pdf","docx","txt","md"],
                                    accept_multiple_files=True, key="jd_uploader")
        if uploaded and st.button("📥 Parse All JDs", type="primary",
                                  use_container_width=True, key="parse_jds_btn"):
            jd_list  = list(st.session_state.get("jd_list", []))
            existing = {j.source_name for j in jd_list}
            added, err_list = 0, []
            with st.spinner("Parsing JDs…"):
                for f in uploaded:
                    if f.name in existing:
                        continue
                    jd = parse_jd_file(f.read(), f.name)
                    if jd:
                        jd_list.append(jd)
                        added += 1
                    else:
                        err_list.append(f.name)
            st.session_state["jd_list"] = jd_list
            alert_box(f"✅ {added} JD(s) parsed and added.", "success")
            for fn in err_list:
                alert_box(f"Could not parse: **{fn}**", "warning")

    with tab_paste:
        jd_text = st.text_area("Paste JD text here", height=260,
                               placeholder="Job Title: Python Backend Engineer\n\nRequired Skills: Python, FastAPI, Docker\nPreferred Skills: AWS, Redis\nMin CGPA: 7.0",
                               key="jd_paste_area")
        jd_name = st.text_input("Role Label / Name", placeholder="e.g. Backend Engineer",
                                key="jd_paste_name")
        if st.button("➕ Add this JD", type="primary", key="add_paste_jd",
                     disabled=not jd_text.strip()):
            jd = parse_job_description(jd_text)
            jd.source_name = jd_name.strip() or "Pasted JD"
            if not jd.role or jd.role == "Not Specified":
                jd.role = jd.source_name
            jds = list(st.session_state.get("jd_list", []))
            jds.append(jd)
            st.session_state["jd_list"] = jds
            alert_box(f"JD **{jd.display_name()}** added.", "success")

    jds = st.session_state.get("jd_list", [])
    if not jds:
        alert_box("No JDs loaded yet. Upload or paste at least one.", "info")
        return

    st.divider()
    section_header(f"✅ Loaded Job Descriptions ({len(jds)})")
    for i, jd in enumerate(jds):
        with st.expander(
            f"**{jd.display_name()}** — {len(jd.required_skills)} required / "
            f"{len(jd.preferred_skills)} preferred skills  |  source: `{jd.source_name}`"
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Role:** {jd.role}")
                st.markdown(f"**Min CGPA:** {jd.min_cgpa or '—'}")
                st.markdown(f"**Degree:** {jd.preferred_degree or '—'}")
                st.markdown(f"**Openings:** {jd.num_slots or '—'}")
                st.markdown(f"**Experience:** {str(jd.experience_years)+' yr(s)' if jd.experience_years else '—'}")
            with c2:
                st.markdown("**Required Skills:**")
                st.markdown(skill_chips(jd.required_skills, "matched"), unsafe_allow_html=True)
                st.markdown("**Preferred Skills:**")
                st.markdown(skill_chips(jd.preferred_skills), unsafe_allow_html=True)
            if st.button(f"🗑️ Remove", key=f"rm_jd_{i}"):
                jds.pop(i)
                st.session_state["jd_list"] = jds
                st.rerun()

    if st.button("🗑️ Clear ALL JDs", key="clear_all_jds"):
        st.session_state["jd_list"] = []
        st.rerun()

    st.divider()
    alert_box("JDs ready. Go to **🔬 Analyze** to run the full pipeline.", "info")

# ── Shared pipeline runner ────────────────────────────────────────────────

def _run_analysis(pdf_files, jd_list, use_semantic):
    from matcher.multi_jd import MultiJDRankingEngine
    engine   = MultiJDRankingEngine(use_semantic=use_semantic)
    p_bar    = st.progress(0.0)
    p_stat   = st.empty()
    allow_mm = st.session_state.get("allow_multi_match", False)

    def on_p(cur, tot, fname):
        p_bar.progress(cur / max(tot, 1))
        p_stat.markdown(f"Parsing `{fname}` ({cur}/{tot})…")

    with st.spinner("Running multi-JD analysis…"):
        results  = engine.process(pdf_files, jd_list, allow_multi_match=allow_mm,
                                  progress_callback=on_p)
        reports  = engine.generate_reports(results, jd_list, allow_multi_match=allow_mm)
        rankings = engine.build_jobwise_rankings(results, jd_list, allow_multi_match=allow_mm)

    p_bar.progress(1.0)
    p_stat.empty()
    st.session_state.update({
        "multi_results":    results,
        "multi_reports":    reports,
        "jobwise_rankings": rankings,
    })
    return results, reports

# ── Page 4: Analyze ───────────────────────────────────────────────────────

def page_analysis():
    from matcher.multi_jd import MultiJDRankingEngine
    section_header("🔬 Analyze")
    pdfs = st.session_state.get("uploaded_pdfs", [])
    jds  = st.session_state.get("jd_list", [])

    if not pdfs:
        alert_box("No resumes loaded. Go to **📂 Upload Resumes**.", "warning"); return
    if not jds:
        alert_box("No JDs loaded. Go to **📄 Upload Job Descriptions**.", "warning"); return

    st.markdown(
        f"**{len(pdfs)} resume(s)** × **{len(jds)} JD(s)** = "
        f"**{len(pdfs)*len(jds)} scoring operations**")

    o1, o2 = st.columns(2)
    with o1:
        use_sem = st.checkbox("Semantic Matching (Sentence Transformers)",
                              st.session_state.get("use_semantic", True), key="an_sem")
    with o2:
        multi_m = st.checkbox("Allow Multi-Match (assign to multiple JDs)",
                              st.session_state.get("allow_multi_match", False), key="an_mm")

    st.session_state["use_semantic"]      = use_sem
    st.session_state["allow_multi_match"] = multi_m

    if st.button("▶️  Run Analysis", type="primary", use_container_width=True, key="run_btn"):
        results, _ = _run_analysis(pdfs, jds, use_sem)
        stats = MultiJDRankingEngine.get_summary_stats(results, jds)
        st.divider()
        section_header("📊 Results Preview")
        cols = st.columns(5)
        for col, (lbl, val, clr) in zip(cols, [
            ("Total",       stats["total"],              PRIMARY_COLOR),
            ("Shortlisted", stats["shortlisted"],        SUCCESS_COLOR),
            ("Reserve",     stats["reserve"],            WARNING_COLOR),
            ("Rejected",    stats["rejected"],           DANGER_COLOR),
            ("Avg Score",   f"{stats['avg_score']:.1f}", INFO_COLOR),
        ]):
            col.markdown(kpi_card(lbl, val, clr), unsafe_allow_html=True)
        fp = sum(1 for r in results if r.parse_result.parse_status == "Failed")
        pc = st.columns(3)
        pc[0].markdown(kpi_card("Parsed OK",    len(results)-fp, SUCCESS_COLOR), unsafe_allow_html=True)
        pc[1].markdown(kpi_card("Parse Failed", fp,              DANGER_COLOR),  unsafe_allow_html=True)
        pc[2].markdown(kpi_card("Success Rate", f"{stats['parse_success_rate']}%", INFO_COLOR), unsafe_allow_html=True)
        alert_box("Done! Go to **🎯 Candidate Best Match** or **🏆 Job-wise Ranking**.", "success")

    prev = st.session_state.get("multi_results", [])
    if prev:
        alert_box(f"Previous run: {len(prev)} candidates. Re-run above to refresh.", "info")

# ── Page 5: Candidate Best Match ─────────────────────────────────────────

def page_best_match():
    from matcher.multi_jd import MultiJDRankingEngine
    section_header("🎯 Candidate Best Match")
    results = st.session_state.get("multi_results", [])
    reports = st.session_state.get("multi_reports", {})
    jds     = st.session_state.get("jd_list", [])

    if not results:
        alert_box("No results. Run **🔬 Analyze** first.", "info"); return

    stats = MultiJDRankingEngine.get_summary_stats(results, jds)
    cols  = st.columns(5)
    for col, (lbl, val, clr) in zip(cols, [
        ("Total",       stats["total"],              PRIMARY_COLOR),
        ("Shortlisted", stats["shortlisted"],        SUCCESS_COLOR),
        ("Reserve",     stats["reserve"],            WARNING_COLOR),
        ("Rejected",    stats["rejected"],           DANGER_COLOR),
        ("Avg Score",   f"{stats['avg_score']:.1f}", INFO_COLOR),
    ]):
        col.markdown(kpi_card(lbl, val, clr), unsafe_allow_html=True)

    st.divider()
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        roles    = ["All"] + sorted({r.best_jd_role for r in results})
        fil_role = st.selectbox("Best Role", roles, key="bm_role")
    with f2:
        fil_st   = st.selectbox("Status", ["All","Shortlisted","Reserve","Rejected"], key="bm_st")
    with f3:
        fil_min  = st.slider("Min Score", 0, 100, 0, key="bm_min")
    with f4:
        fil_cgpa = st.slider("Min CGPA", 0.0, 10.0, 0.0, 0.1, key="bm_cgpa")

    filtered = [r for r in results
                if (fil_role == "All" or r.best_jd_role == fil_role)
                and (fil_st == "All" or r.best_shortlist == fil_st)
                and r.best_score >= fil_min
                and (fil_cgpa == 0 or (r.candidate.normalized_cgpa or 0) >= fil_cgpa)]

    st.markdown(f"**Showing {len(filtered)} of {len(results)} candidates**")

    rows = []
    for rank, r in enumerate(sorted(filtered, key=lambda x: x.best_score, reverse=True), 1):
        best_jd = next((s for s in r.jd_scores if s.jd_role == r.best_jd_role), None)
        rows.append({
            "Rank":      rank,
            "Name":      r.name,
            "Best Role": r.best_jd_role,
            "Score":     f"{r.best_score:.1f}",
            "Status":    f"{status_badge(r.best_shortlist)} {r.best_shortlist}",
            "Conf":      f"{confidence_badge(r.confidence)} {r.confidence}",
            "CGPA":      f"{r.candidate.normalized_cgpa:.2f}" if r.candidate.normalized_cgpa else "N/A",
            "Matched":   ", ".join((best_jd.matched_required if best_jd else [])[:4]),
            "Missing":   ", ".join((best_jd.missing_required if best_jd else [])[:3]),
            "Email":     r.candidate.email or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    section_header("🗂️ Candidate Detail Cards")
    for r in sorted(filtered, key=lambda x: x.best_score, reverse=True):
        best_jd = next((s for s in r.jd_scores if s.jd_role == r.best_jd_role), None)
        border  = (SUCCESS_COLOR if r.best_shortlist=="Shortlisted" else
                   WARNING_COLOR if r.best_shortlist=="Reserve" else DANGER_COLOR)
        with st.expander(
            f"{status_badge(r.best_shortlist)} **{r.name}** — "
            f"{r.best_score:.1f}/100 | {r.best_jd_role} | "
            f"{confidence_badge(r.confidence)} {r.confidence}", expanded=False):

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                st.markdown("**Contact & Education**")
                st.markdown(f"📧 {r.candidate.email or '—'}")
                st.markdown(f"📱 {r.candidate.phone or '—'}")
                st.markdown(f"🎓 {r.candidate.college or '—'}")
                st.markdown(f"📚 {r.candidate.degree} {r.candidate.branch or ''}")
                if r.candidate.graduation_year:
                    st.markdown(f"🗓️ Grad: {r.candidate.graduation_year}")
                if r.candidate.normalized_cgpa:
                    st.markdown(f"⭐ CGPA: {r.candidate.normalized_cgpa:.2f}/10")
                if r.candidate.github:
                    st.markdown(f"🐙 [GitHub]({r.candidate.github})")
                if r.candidate.linkedin:
                    st.markdown(f"💼 [LinkedIn]({r.candidate.linkedin})")

            with c2:
                st.markdown("**Scores vs All JDs**")
                for js in sorted(r.jd_scores, key=lambda s: s.score, reverse=True):
                    is_best = "⭐ " if js.jd_role == r.best_jd_role else ""
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;margin-bottom:4px;">'
                        f'<span style="font-size:.82rem;">{is_best}{js.jd_role}</span>'
                        f'<span style="color:{score_color(js.score)};font-weight:700;">{js.score:.1f}%</span>'
                        f'</div>', unsafe_allow_html=True)
                    st.progress(js.score / 100)

            with c3:
                st.markdown("**Parse Info**")
                st.markdown(f"Status: `{r.parse_result.parse_status}`")
                st.markdown(f"Words: `{r.parse_result.word_count}`")
                st.markdown(f"Via: `{r.parse_result.strategy_used}`")
                st.markdown(f"Fields: `{r.confidence_report.completeness_pct}%`")

            if best_jd:
                st.markdown("**Matched Required Skills:**")
                st.markdown(skill_chips(best_jd.matched_required, "matched"), unsafe_allow_html=True)
                st.markdown("**Missing Required Skills:**")
                st.markdown(skill_chips(best_jd.missing_required, "missing"), unsafe_allow_html=True)
                st.markdown("**Top 3 Reasons:**")
                for i, reason in enumerate(best_jd.reasons, 1):
                    st.markdown(f"{i}. {reason}")

            if r.recommendations:
                st.markdown("---")
                st.markdown("**🤖 AI Role Recommendations** *(below threshold on all JDs)*")
                for rec in r.recommendations:
                    st.markdown(
                        f'<div style="background:#1a1f2e;border-left:3px solid {INFO_COLOR};'
                        f'padding:8px 12px;border-radius:6px;margin-bottom:5px;">'
                        f'<strong style="color:{INFO_COLOR};">{rec.role}</strong> — '
                        f'<span style="color:{score_color(rec.match_pct)};">{rec.match_pct:.0f}%</span> match<br>'
                        f'<span style="color:#8892b0;font-size:.82rem;">{rec.reason}</span></div>',
                        unsafe_allow_html=True)

    st.divider()
    _dl_btn("⬇️ Download candidate_best_match.csv", reports.get("candidate_best_match"),
            "candidate_best_match.csv")

# ── Page 6: Job-wise Ranking ──────────────────────────────────────────────

def page_job_ranking():
    from matcher.ranking import classify_candidate
    section_header("🏆 Job-wise Ranking")
    rankings = st.session_state.get("jobwise_rankings", {})
    reports  = st.session_state.get("multi_reports", {})

    if not rankings:
        alert_box("No rankings yet. Run **🔬 Analyze** first.", "info"); return

    jd_tabs = st.tabs([f"📋 {role}" for role in rankings])
    for tab, (role, rows) in zip(jd_tabs, rankings.items()):
        with tab:
            if not rows:
                st.info(f"No candidates scored for **{role}**.")
                continue
            sl = sum(1 for r in rows if r["shortlist_status"] == "Shortlisted")
            rv = sum(1 for r in rows if r["shortlist_status"] == "Reserve")
            rj = sum(1 for r in rows if r["shortlist_status"] == "Rejected")
            kc = st.columns(4)
            kc[0].markdown(kpi_card("Candidates", len(rows), PRIMARY_COLOR), unsafe_allow_html=True)
            kc[1].markdown(kpi_card("Shortlisted", sl,       SUCCESS_COLOR), unsafe_allow_html=True)
            kc[2].markdown(kpi_card("Reserve",     rv,       WARNING_COLOR), unsafe_allow_html=True)
            kc[3].markdown(kpi_card("Rejected",    rj,       DANGER_COLOR),  unsafe_allow_html=True)

            fc1, fc2, fc3 = st.columns(3)
            with fc1: fil_st  = st.selectbox("Status", ["All","Shortlisted","Reserve","Rejected"],
                                              key=f"jwr_st_{role}")
            with fc2: fil_min = st.slider("Min Score", 0, 100, 0, key=f"jwr_min_{role}")
            with fc3: fil_sk  = st.text_input("Must-have skill", key=f"jwr_sk_{role}",
                                               placeholder="e.g. Python")
            vis = [r for r in rows if
                   (fil_st == "All" or r["shortlist_status"] == fil_st) and
                   r["score"] >= fil_min and
                   (not fil_sk.strip() or fil_sk.strip().lower() in r.get("matched_required","").lower())]

            st.markdown(f"**{len(vis)} candidate(s)**")
            display = [{
                "Rank":    r["rank"],
                "Name":    r["name"],
                "Score":   r["score"],
                "Status":  f"{status_badge(r['shortlist_status'])} {r['shortlist_status']}",
                "CGPA":    f"{r['cgpa']:.2f}" if r.get("cgpa") else "N/A",
                "Matched": r.get("matched_required","")[:60],
                "Missing": r.get("missing_required","")[:40],
                "Email":   r.get("email","—"),
            } for r in vis]
            st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)

            st.markdown("#### 🥇 Top 3")
            for r in vis[:3]:
                border = (SUCCESS_COLOR if r["shortlist_status"]=="Shortlisted" else
                          WARNING_COLOR if r["shortlist_status"]=="Reserve" else DANGER_COLOR)
                st.markdown(
                    f'<div style="background:#1a1f2e;border-left:4px solid {border};'
                    f'padding:10px 14px;border-radius:8px;margin-bottom:7px;">'
                    f'<strong>#{r["rank"]} {r["name"]}</strong> — '
                    f'<span style="color:{score_color(r["score"])};font-weight:700;">'
                    f'{r["score"]:.1f}/100</span> | '
                    f'{status_badge(r["shortlist_status"])} {r["shortlist_status"]}<br>'
                    f'<span style="color:#8892b0;font-size:.82rem;">'
                    f'Matched: {r.get("matched_required","—")[:60]}</span></div>',
                    unsafe_allow_html=True)

    st.divider()
    _dl_btn("⬇️ Download job_wise_ranking.csv", reports.get("job_wise_ranking"),
            "job_wise_ranking.csv")

# ── Page 7: Analytics ────────────────────────────────────────────────────

def page_analytics():
    section_header("📈 Analytics Dashboard")
    results  = st.session_state.get("multi_results", [])
    rankings = st.session_state.get("jobwise_rankings", {})
    reports  = st.session_state.get("multi_reports", {})

    if not results:
        alert_box("No results yet. Run **🔬 Analyze** first.", "info"); return

    scores   = [r.best_score for r in results]
    statuses = [r.best_shortlist for r in results]
    cgpas    = [r.candidate.normalized_cgpa for r in results if r.candidate.normalized_cgpa]
    parse_st = [r.parse_result.parse_status for r in results]

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### 👥 Candidates per Job")
        jc = {role: len(rows) for role, rows in rankings.items()}
        if jc:
            fig = px.bar(x=list(jc.keys()), y=list(jc.values()),
                         color=list(jc.values()),
                         color_continuous_scale=[[0,"#252b3e"],[1,PRIMARY_COLOR]],
                         text_auto=True, labels={"x":"Role","y":"Count"})
            fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                              xaxis=dict(gridcolor="#2a2f3e"), yaxis=dict(gridcolor="#2a2f3e"))
            _plotly_dark(fig); st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown("#### 📊 Score Distribution")
        fig = px.histogram(x=scores, nbins=10, color_discrete_sequence=[PRIMARY_COLOR],
                           labels={"x":"Score","y":"Count"})
        fig.add_vline(x=SHORTLIST_THRESHOLD, line_dash="dash",
                      line_color=SUCCESS_COLOR, annotation_text="Shortlist")
        fig.add_vline(x=RESERVE_THRESHOLD, line_dash="dash",
                      line_color=WARNING_COLOR, annotation_text="Reserve")
        fig.update_layout(height=320, xaxis=dict(gridcolor="#2a2f3e"), yaxis=dict(gridcolor="#2a2f3e"))
        _plotly_dark(fig); st.plotly_chart(fig, use_container_width=True)

    r3, r4 = st.columns(2)
    with r3:
        st.markdown("#### 🎯 Avg Score per Job")
        avg = {role: round(sum(r["score"] for r in rows)/len(rows),1)
               for role, rows in rankings.items() if rows}
        if avg:
            fig = px.bar(x=list(avg.values()), y=list(avg.keys()),
                         orientation="h",
                         color=list(avg.values()),
                         color_continuous_scale=[[0,DANGER_COLOR],[0.5,WARNING_COLOR],[1,SUCCESS_COLOR]],
                         text_auto=True, labels={"x":"Avg Score","y":"Role"})
            fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed", gridcolor="#2a2f3e"),
                              xaxis=dict(range=[0,100], gridcolor="#2a2f3e"))
            _plotly_dark(fig); st.plotly_chart(fig, use_container_width=True)

    with r4:
        st.markdown("#### 🥧 Shortlist Distribution")
        sc = {}
        for s in statuses: sc[s] = sc.get(s,0)+1
        fig = px.pie(names=list(sc.keys()), values=list(sc.values()), hole=0.4,
                     color=list(sc.keys()),
                     color_discrete_map={"Shortlisted":SUCCESS_COLOR,
                                         "Reserve":WARNING_COLOR,"Rejected":DANGER_COLOR})
        fig.update_layout(height=320)
        _plotly_dark(fig); st.plotly_chart(fig, use_container_width=True)

    r5, r6 = st.columns(2)
    with r5:
        st.markdown("#### 🛠️ Top 15 Skills")
        sf: dict[str,int] = {}
        for r in results:
            for sk in r.candidate.skills: sf[sk] = sf.get(sk,0)+1
        top = sorted(sf.items(), key=lambda x:-x[1])[:15]
        if top:
            fig = px.bar(x=[t[1] for t in top], y=[t[0] for t in top],
                         orientation="h",
                         color=[t[1] for t in top],
                         color_continuous_scale=[[0,"#252b3e"],[1,PRIMARY_COLOR]],
                         text_auto=True, labels={"x":"Count","y":"Skill"})
            fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed", gridcolor="#2a2f3e"),
                              xaxis=dict(gridcolor="#2a2f3e"))
            _plotly_dark(fig); st.plotly_chart(fig, use_container_width=True)

    with r6:
        st.markdown("#### 🎓 CGPA Distribution")
        if cgpas:
            fig = px.histogram(x=cgpas, nbins=10, color_discrete_sequence=[INFO_COLOR],
                               labels={"x":"CGPA (10-pt)","y":"Count"})
            fig.update_layout(height=420, xaxis=dict(range=[0,10], gridcolor="#2a2f3e"),
                              yaxis=dict(gridcolor="#2a2f3e"))
            _plotly_dark(fig); st.plotly_chart(fig, use_container_width=True)
        else:
            alert_box("CGPA data not available.", "warning")

    r7, r8 = st.columns(2)
    with r7:
        st.markdown("#### 🔍 Parse Quality")
        pc: dict[str,int] = {}
        for s in parse_st: pc[s] = pc.get(s,0)+1
        fig = px.bar(x=list(pc.keys()), y=list(pc.values()),
                     color=list(pc.keys()),
                     color_discrete_map={"Clean":SUCCESS_COLOR,"Partial":INFO_COLOR,
                                         "OCR":WARNING_COLOR,"Failed":DANGER_COLOR},
                     text_auto=True)
        fig.update_layout(height=300, showlegend=False,
                          xaxis=dict(gridcolor="#2a2f3e"), yaxis=dict(gridcolor="#2a2f3e"))
        _plotly_dark(fig); st.plotly_chart(fig, use_container_width=True)

    with r8:
        st.markdown("#### ❌ Top Missing Skills")
        miss: dict[str,int] = {}
        for r in results:
            best = next((s for s in r.jd_scores if s.jd_role == r.best_jd_role), None)
            if best:
                for sk in best.missing_required: miss[sk] = miss.get(sk,0)+1
        top_miss = sorted(miss.items(), key=lambda x:-x[1])[:12]
        if top_miss:
            fig = px.bar(x=[t[1] for t in top_miss], y=[t[0] for t in top_miss],
                         orientation="h", color_discrete_sequence=[DANGER_COLOR],
                         text_auto=True, labels={"x":"Count","y":"Skill"})
            fig.update_layout(height=300,
                              yaxis=dict(autorange="reversed", gridcolor="#2a2f3e"),
                              xaxis=dict(gridcolor="#2a2f3e"))
            _plotly_dark(fig); st.plotly_chart(fig, use_container_width=True)

    st.divider()
    _dl_btn("⬇️ Download analytics.csv", reports.get("analytics"), "analytics.csv")

# ── Page 8: Reports ───────────────────────────────────────────────────────

def page_reports():
    section_header("📑 Reports & Export")
    reports = st.session_state.get("multi_reports", {})
    if not reports:
        alert_box("No reports yet. Run **🔬 Analyze** first.", "info"); return

    st.markdown("All reports are auto-saved to `data/outputs/`. Download individually below.")
    st.divider()

    defs = [
        ("candidate_best_match","🎯 Candidate Best Match",   "candidate_best_match.csv",
         "Each candidate with their best-matching JD, score, matched/missing skills, reasons."),
        ("job_wise_ranking",    "🏆 Job-wise Ranking",        "job_wise_ranking.csv",
         "All candidates ranked per job role."),
        ("shortlisted",         "✅ Shortlisted",              "shortlisted.csv",
         "Only shortlisted candidates (score ≥ 70)."),
        ("reserve",             "🟡 Reserve",                  "reserve.csv",
         "Reserve candidates (score 50–69)."),
        ("rejected",            "❌ Rejected",                  "rejected.csv",
         "Rejected candidates (score < 50)."),
        ("parse_quality",       "🔍 Parse Quality Report",    "parse_quality_report.csv",
         "Per-resume parse status, strategy, word count, field completeness."),
        ("analytics",           "📈 Analytics",                "analytics.csv",
         "Skill frequency, score buckets, candidates per job, status counts."),
    ]

    for key, label, fname, desc in defs:
        df = reports.get(key)
        c1, c2 = st.columns([3, 1])
        with c1:
            cnt = len(df) if df is not None and not df.empty else 0
            st.markdown(
                f'<div style="background:#1a1f2e;border-radius:8px;padding:12px 15px;'
                f'border:1px solid #2a2f3e;margin-bottom:6px;">'
                f'<strong style="color:#cdd;">{label}</strong> '
                f'<code style="font-size:.77rem;color:#8892b0;">{fname}</code> — {cnt} rows<br>'
                f'<span style="color:#8892b0;font-size:.81rem;">{desc}</span></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            _dl_btn(f"⬇️ {fname}", df, fname)

    st.divider()
    if st.button("📦 Download All Combined", use_container_width=True, key="dl_combined"):
        frames = []
        for key, _, fname, _ in defs:
            df = reports.get(key)
            if df is not None and not df.empty:
                tmp = df.copy(); tmp.insert(0, "_report", key)
                frames.append(tmp)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            st.download_button("⬇️ combined_reports.csv",
                               df_to_csv_bytes(combined),
                               "combined_reports.csv", "text/csv",
                               use_container_width=True)

# ── Page 9: Settings ──────────────────────────────────────────────────────

def page_settings():
    section_header("⚙️ Settings")
    st.markdown("Tune matching behaviour and manage session data.")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        section_header("🎯 Shortlisting Thresholds")
        st.info(f"Current: Shortlisted ≥ **{SHORTLIST_THRESHOLD}**  |  Reserve ≥ **{RESERVE_THRESHOLD}**")
        st.caption("To change thresholds permanently, edit `config.py`.")

        section_header("🔀 Matching Options")
        st.session_state["allow_multi_match"] = st.checkbox(
            "Allow Multi-Match (assign candidate to multiple JDs)",
            st.session_state.get("allow_multi_match", False), key="sett_mm")
        st.session_state["use_semantic"] = st.checkbox(
            "Enable Semantic Matching (Sentence Transformers)",
            st.session_state.get("use_semantic", True), key="sett_sem")

    with c2:
        section_header("⚖️ Scoring Weights")
        for comp, wt in SCORING_WEIGHTS.items():
            progress_row(comp.replace("_"," ").title(), wt)

        section_header("📂 Output Paths")
        from config import OUTPUT_DIR, TEMP_RESUME_DIR
        st.code(f"Reports:   {OUTPUT_DIR}\nTemp PDFs: {TEMP_RESUME_DIR}")
        if st.button("🗑️ Clear temp resumes", key="clear_tmp"):
            import shutil
            try:
                shutil.rmtree(str(TEMP_RESUME_DIR))
                TEMP_RESUME_DIR.mkdir(parents=True, exist_ok=True)
                alert_box("Temp folder cleared.", "success")
            except Exception as exc:
                alert_box(f"Error: {exc}", "error")

    st.divider()
    if st.button("🔄 Reset ALL session data", key="reset_all"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Page 10: About ────────────────────────────────────────────────────────

def page_about():
    section_header("ℹ️ About InternLoom AI")
    st.markdown(f"""
**InternLoom AI v{APP_VERSION}** — Intelligent Multi-JD Resume Shortlisting Engine.

| Detail | Info |
|---|---|
| Version | `{APP_VERSION}` |
| Python | 3.12+ |
| UI | Streamlit |
| PDF Parsing | PyMuPDF · pdfplumber · pdfminer.six |
| OCR | EasyOCR + OpenCV |
| NLP | spaCy en_core_web_sm |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Fuzzy Match | RapidFuzz |
| Drive Import | gdown + requests (by File ID, %PDF verified) |
| Reports | 7 CSV files |
""")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        section_header("🆕 v2 Features")
        for item in [
            "Fixed Google Drive import (gdown, File-ID based, %PDF verify, retry)",
            "Multi-JD upload: PDF, DOCX, TXT formats",
            "Every resume scored vs every JD automatically",
            "Job-wise ranking tabs with per-role filters",
            "Candidate Best Match page — all JD scores visible",
            "AI role recommendation for low-scoring candidates",
            "7 CSV export files including shortlisted/reserve/rejected",
            "Settings page for thresholds and options",
        ]:
            st.markdown(f"✅ {item}")
    with c2:
        section_header("📁 Output Files")
        for f, d in [
            ("candidate_best_match.csv","Best role + score per candidate"),
            ("job_wise_ranking.csv",    "Ranked lists per job role"),
            ("shortlisted.csv",         "Score ≥ 70"),
            ("reserve.csv",             "Score 50–69"),
            ("rejected.csv",            "Score < 50"),
            ("parse_quality_report.csv","Per-resume parse metadata"),
            ("analytics.csv",           "Aggregated metrics data"),
        ]:
            st.markdown(f"**`{f}`** — {d}")
    st.divider()
    st.caption("All processing is 100% local. No resume data is sent to external servers.")

# ── Router + Entry point ──────────────────────────────────────────────────

PAGE_ROUTER = {
    "home":          page_home,
    "upload_resumes":page_upload_resumes,
    "upload_jd":     page_upload_jd,
    "analysis":      page_analysis,
    "best_match":    page_best_match,
    "job_ranking":   page_job_ranking,
    "analytics":     page_analytics,
    "reports":       page_reports,
    "settings":      page_settings,
    "about":         page_about,
}

def run_dashboard():
    st.set_page_config(
        page_title=f"{APP_TITLE} — {APP_SUBTITLE}",
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": f"**{APP_TITLE}** v{APP_VERSION}"},
    )
    inject_css()
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    selected = render_sidebar()
    PAGE_ROUTER.get(selected, page_home)()
