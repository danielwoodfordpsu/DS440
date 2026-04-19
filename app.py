import os
import re
from datetime import date, datetime
from typing import Any

import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client


_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_APP_DIR, "assets")


def get_mascot_path() -> str | None:
    for fname in ("dr_opak_mascot.webp", "dr_opak_mascot.png"):
        p = os.path.join(_ASSETS_DIR, fname)
        if os.path.isfile(p):
            return p
    return None


load_dotenv()

st.set_page_config(page_title="Dr Opak's Fitness Lab", page_icon="🩺", layout="wide")


def inject_dr_opak_theme_light() -> None:
    """High-contrast Dr Opak UI: white panels, black type. Canvas stays light gray."""
    st.markdown(
        """
<style>
  .stApp {
    background-color: #d0d0d0 !important;
    color: #000000 !important;
  }
  [data-testid="stAppViewContainer"] {
    color: #000000 !important;
  }
  [data-testid="stHeader"] {
    background-color: #ffffff !important;
    border-bottom: 1px solid #000000;
  }
  section[data-testid="stMain"] > div {
    background-color: #ffffff !important;
  }
  .block-container {
    background-color: #ffffff !important;
    color: #000000 !important;
  }
  [data-testid="stSidebarContent"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-right: 2px solid #000000;
  }
  h1, h2, h3, h4, h5, h6 {
    font-family: Georgia, "Times New Roman", serif !important;
    color: #000000 !important;
  }
  .stApp p, .stApp span, .stApp label, .stApp li, .stApp small,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] span,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] label,
  .stCaption, span[data-testid="stCaption"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
  }
  .dr-opak-tagline {
    font-family: Georgia, "Times New Roman", serif;
    color: #000000 !important;
    font-style: italic;
    border-left: 4px solid #000000;
    padding-left: 10px;
    margin: 0.25rem 0 0.75rem 0;
  }
  /* Form controls: white fields, black text */
  .stTextInput input, .stTextArea textarea,
  [data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
  .stNumberInput input, [data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border: 1px solid #000000 !important;
  }
  [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border: 1px solid #000000 !important;
  }
  [data-baseweb="select"] span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
  }
  .stSlider label, .stSlider span, .stCheckbox label, .stRadio label {
    color: #000000 !important;
  }
  .stButton > button,
  [data-testid="stFormSubmitButton"] button {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border: 2px solid #000000 !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    background-color: #ffffff !important;
  }
  div[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #000000 !important;
  }
  .streamlit-expanderHeader {
    background-color: #ffffff !important;
    color: #000000 !important;
  }
  [data-testid="stAlert"] {
    color: #000000 !important;
  }
  [data-testid="stAlert"] p, [data-testid="stAlert"] span {
    color: #000000 !important;
  }
  [data-testid="stDataFrame"] {
    background-color: #ffffff !important;
  }
  .stApp code, .stApp pre {
    background-color: #f5f5f5 !important;
    color: #000000 !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )


inject_dr_opak_theme_light()


def render_dr_opak_banner() -> None:
    mascot = get_mascot_path()
    if mascot:
        c1, c2 = st.columns([0.18, 0.82])
        with c1:
            st.image(mascot, width=110)
        with c2:
            st.markdown("## Dr Opak's Fitness Laboratory")
            st.markdown(
                '<p class="dr-opak-tagline">Good news, everyone! Science, sets, and slightly '
                "unhinged spotting advice await.</p>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("## Dr Opak's Fitness Laboratory")
        st.markdown(
            '<p class="dr-opak-tagline">Good news, everyone! Drop your keys in the chalk bucket.</p>',
            unsafe_allow_html=True,
        )


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_supabase_client() -> Client | None:
    url = _env("SUPABASE_URL")
    key = _env("SUPABASE_ANON_KEY") or _env("SUPABASE_PUBLISHABLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_qdrant_client() -> QdrantClient | None:
    url = _env("QDRANT_URL").strip().rstrip("/")
    api_key = _env("QDRANT_API_KEY")
    if not url or not api_key:
        return None
    # Strip accidental /collections suffix (common copy-paste mistake → 404)
    for bad in ("/collections", "/dashboard"):
        if url.lower().endswith(bad):
            url = url[: -len(bad)].rstrip("/")
    return QdrantClient(url=url, api_key=api_key, timeout=120)


def init_gemini() -> bool:
    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    return True


def supabase_insert_lift(supabase: Client, lift: dict[str, Any]) -> None:
    supabase.table("lifts").insert(lift).execute()


def supabase_fetch_lifts(supabase: Client, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    query = (
        supabase.table("lifts")
        .select("*")
        .eq("user_id", user_id)
        .order("lifted_at", desc=True)
        .limit(limit)
    )
    response = query.execute()
    return response.data or []


def format_lifts_for_llm(rows: list[dict[str, Any]], athlete_id: str) -> str:
    if not rows:
        return f"No lift rows found for user_id `{athlete_id}`."
    lines: list[str] = []
    for i, r in enumerate(rows, start=1):
        ex = r.get("exercise", "")
        w = r.get("weight", "")
        rep = r.get("reps", "")
        sets_n = r.get("sets", "")
        rpe = r.get("rpe")
        when = r.get("lifted_at", "")
        notes = (r.get("notes") or "").replace("\n", " ").strip()
        if len(notes) > 140:
            notes = notes[:137] + "..."
        rpe_part = f" RPE {rpe}" if rpe is not None and str(rpe).strip() != "" else ""
        note_part = f" | notes: {notes}" if notes else ""
        lines.append(f"{i}. {when} | {ex} | {w} x {rep} x {sets_n}{rpe_part}{note_part}")
    return "\n".join(lines)


def lift_log_prompt_section(lift_context: str | None) -> str:
    if not lift_context:
        return ""
    return f"""
### Athlete lift log (from your app database)
User-supplied training history (not published research). Reference naturally; do **not** use [Source N] tags for these rows — those are only for paper excerpts.

{lift_context}

"""


@st.cache_resource
def get_local_embedding_model() -> SentenceTransformer:
    model_name = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def embed_text(text: str) -> list[float]:
    # Match the user's working local RAG approach (MiniLM sentence-transformers).
    model = get_local_embedding_model()
    return model.encode(text).tolist()


def _paper_dedupe_key(chunk: dict[str, Any]) -> str:
    """Stable key so multiple chunks from the same paper collapse to one source."""
    doi = (chunk.get("doi") or "").strip()
    if doi:
        return doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "")
    url = (chunk.get("pubmed_url") or "").strip()
    if url:
        return url
    pmid = (chunk.get("pmid") or "").strip()
    if pmid.startswith("http"):
        return pmid
    title = (chunk.get("title") or "").strip().lower()
    year = str(chunk.get("year") or "")
    return f"{title}|{year}"


def dedupe_chunks_keep_best_score(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        key = _paper_dedupe_key(chunk)
        if not key or key == "|":
            key = f"anon_{id(chunk)}"
        prev = best.get(key)
        if prev is None or chunk["score"] > prev["score"]:
            best[key] = chunk
    return sorted(best.values(), key=lambda c: -c["score"])


def qdrant_search_context(
    qdrant: QdrantClient,
    collection_name: str,
    question: str,
    top_k: int = 5,
    focus_topic: str | None = None,
    quality_min: int = 0,
    year_from: int = 0,
) -> list[dict[str, Any]]:
    vector = embed_text(question)
    query_filter = None
    if focus_topic:
        query_filter = Filter(
            must=[FieldCondition(key="topic", match=MatchValue(value=focus_topic))]
        )

    # Over-fetch then filter + dedupe (same idea as your local RAG script).
    fetch_limit = min(max(top_k * 5, 40), 200)

    response = qdrant.query_points(
        collection_name=collection_name,
        query=vector,
        limit=fetch_limit,
        query_filter=query_filter,
        with_payload=True,
    )
    hits = response.points
    results: list[dict[str, Any]] = []
    for item in hits:
        payload = item.payload or {}
        row = {
            "score": float(item.score),
            "text": payload.get("text", ""),
            "source": payload.get("source", "unknown"),
            "topic": payload.get("topic", ""),
            "title": payload.get("title", ""),
            "year": payload.get("year", ""),
            "doi": payload.get("doi", ""),
            "pubmed_url": payload.get("pubmed_url", ""),
            "pmid": payload.get("pmid", ""),
            "authors": payload.get("authors", ""),
            "journal": payload.get("journal", ""),
            "quality_score": int(payload.get("quality_score") or 0),
        }
        if quality_min and row["quality_score"] < quality_min:
            continue
        y = row["year"]
        try:
            if year_from and y:
                y_int = int(str(y).strip()[:4])
                if y_int < year_from:
                    continue
        except (ValueError, TypeError):
            pass
        results.append(row)

    deduped = dedupe_chunks_keep_best_score(results)
    return deduped[:top_k]


def answer_with_context(
    question: str, context_chunks: list[dict[str, Any]], lift_context: str | None = None
) -> str:
    context_block = "\n\n".join(
        [
            (
                f"[Source {idx}]\n"
                f"Title: {chunk['title']}\n"
                f"Authors: {chunk['authors']}\n"
                f"Journal: {chunk['journal']}\n"
                f"Year: {chunk['year']}\n"
                f"DOI: {chunk['doi']}\n"
                f"PubMed URL: {chunk['pubmed_url']}\n"
                f"PMID: {chunk['pmid']}\n"
                f"Source: {chunk['source']}\n"
                f"Topic: {chunk['topic']}\n"
                f"Quality: {chunk['quality_score']}\n"
                f"Content: {chunk['text'][:1500]}"
            )
            for idx, chunk in enumerate(context_chunks, start=1)
        ]
    )
    prompt = f"""
You are **Dr Opak** — a warm, eccentric professor-energy fitness coach who still respects evidence.
Ground research claims in the retrieved papers below. If a lift log is included, use it to personalize programming (volume, frequency, recovery) but keep [Source N] citations for **paper excerpts only**.

Evidence & citations:
- Use ONLY the retrieved paper excerpts for research factual claims (methods, findings, numbers, quotes). If something is not in the excerpts, say so briefly or omit it.
- When you state a finding from the papers, add an inline citation: [Source N] or [Source N, Source M] at the end of the sentence/block it supports.
- Merge overlapping points; prefer **fewer, richer** points over many one-liners.

Depth & structure:
- Write in **markdown**. Short intro, then `###` subsections as fits the question.
- 2–5 sentences per theme where useful; end with **Practical takeaway** bullets (cite papers where relevant).

Do NOT add a "Sources" or "References" section — the app attaches links.
Do NOT print a top-level heading that says only "Answer" — you may use `###` for real section titles.

Question:
{question}
{lift_log_prompt_section(lift_context)}
Retrieved context:
{context_block}
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.55,
            "top_p": 0.95,
            "max_output_tokens": 2048,
        },
    )
    return response.text or "I could not generate an answer."


def citation_url(chunk: dict[str, Any]) -> str:
    doi = (chunk.get("doi") or "").strip()
    pubmed = (chunk.get("pubmed_url") or "").strip()
    pmid = (chunk.get("pmid") or "").strip()
    if doi:
        if doi.startswith("http"):
            return doi
        slug = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").lstrip("/")
        return f"https://doi.org/{slug}"
    if pubmed.startswith("http"):
        return pubmed
    if pmid.startswith("http"):
        return pmid
    return ""


def build_sources_markdown(context_chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(context_chunks, start=1):
        title = (chunk.get("title") or "Untitled paper").strip()
        year = chunk.get("year") or "n.d."
        link = citation_url(chunk)
        label = f"{title} ({year})"
        if link:
            lines.append(f"{idx}. [{label}]({link})")
        else:
            lines.append(f"{idx}. {label}")
    return "\n".join(lines)


def strip_model_sources_section(text: str) -> str:
    """Remove trailing 'Sources' blocks that only list [Source N] (we show References in UI)."""
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if re.match(r"^#{0,3}\s*Sources\s*$", stripped, re.I):
            rest = lines[i + 1 :]
            if _lines_look_like_source_index_only(rest):
                return "\n".join(lines[:i]).strip()
        if stripped.lower() == "sources":
            rest = lines[i + 1 :]
            if _lines_look_like_source_index_only(rest):
                return "\n".join(lines[:i]).strip()
    return text


def _lines_look_like_source_index_only(rest: list[str]) -> bool:
    if not rest:
        return False
    nonempty = [ln for ln in rest if ln.strip()]
    if not nonempty:
        return False
    for ln in nonempty:
        s = ln.strip()
        if not re.match(r"^(\[Source \d+\](\s*,\s*\[Source \d+\])*)$", s):
            return False
    return True


def strip_leading_answer_heading(text: str) -> str:
    lines = text.splitlines()
    while lines and lines[0].strip().lower() in {"answer", "# answer", "## answer", "### answer"}:
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def postprocess_rag_answer_body(text: str) -> str:
    t = strip_leading_answer_heading(text.strip())
    t = strip_model_sources_section(t)
    return t.strip()


def ensure_inline_source_hint(answer: str) -> str:
    if "[Source " in answer:
        return answer
    return (
        answer
        + "\n\n"
        + "_Note: Inline [Source N] tags were not found. See **References** below for the papers used in retrieval._"
    )


def answer_without_context(question: str, lift_context: str | None = None) -> str:
    prompt = f"""
You are **Dr Opak** — a helpful, enthusiastic strength coach with old-school bedside manner.
Give concise, safe, practical advice. If a lift log is included, personalize recommendations to that history.
If needed, mention uncertainty and suggest consulting a professional for injuries or medical concerns.

Question:
{question}
{lift_log_prompt_section(lift_context)}
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text or "I could not generate an answer."


def render_env_status() -> None:
    mascot = get_mascot_path()
    if mascot:
        st.sidebar.image(mascot, width=120)
    st.sidebar.markdown("### Dr Opak")
    st.sidebar.caption("Ph.D. in Heavy Things — *Good news, everyone!*")
    st.sidebar.markdown("---")
    st.sidebar.header("Connections")
    checks = {
        "Supabase URL": bool(_env("SUPABASE_URL")),
        "Supabase Key": bool(_env("SUPABASE_ANON_KEY") or _env("SUPABASE_PUBLISHABLE_KEY")),
        "Qdrant URL": bool(_env("QDRANT_URL")),
        "Qdrant API Key": bool(_env("QDRANT_API_KEY")),
        "Gemini API Key": bool(_env("GEMINI_API_KEY")),
    }
    for label, ok in checks.items():
        st.sidebar.write(f"{'✅' if ok else '❌'} {label}")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Set env vars in `.env` before use. See README for table/collection setup."
    )


def render_lifts_tab(supabase: Client | None) -> None:
    st.subheader("Dr Opak's lift ledger")
    if not supabase:
        st.warning("Connect Supabase first (`SUPABASE_URL` + `SUPABASE_ANON_KEY`).")
        return

    with st.form("lift_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            user_id = st.text_input("User ID", placeholder="athlete_123")
            exercise = st.text_input("Exercise", placeholder="Back Squat")
            lifted_on = st.date_input("Date", value=date.today())
        with col2:
            weight = st.number_input("Weight", min_value=0.0, value=100.0, step=2.5)
            reps = st.number_input("Reps", min_value=1, value=5, step=1)
            sets = st.number_input("Sets", min_value=1, value=3, step=1)
        with col3:
            notes = st.text_area("Notes", placeholder="RPE 8, felt strong.")

        submitted = st.form_submit_button("Save Lift")

    if submitted:
        if not user_id or not exercise:
            st.error("`User ID` and `Exercise` are required.")
        else:
            lift_payload = {
                "user_id": user_id,
                "exercise": exercise,
                "weight": float(weight),
                "reps": int(reps),
                "sets": int(sets),
                "notes": notes,
                "lifted_at": datetime.combine(lifted_on, datetime.min.time()).isoformat(),
            }
            try:
                supabase_insert_lift(supabase, lift_payload)
                st.success("Lift saved.")
            except Exception as exc:
                st.error(f"Could not save lift: {exc}")

    st.markdown("---")
    st.subheader("Recent Lift History")
    history_user = st.text_input(
        "Load history for User ID", placeholder="athlete_123", key="history_user"
    )
    limit = st.slider("Rows", min_value=10, max_value=200, value=50, step=10)

    if st.button("Load Lifts"):
        if not history_user:
            st.info("Enter a User ID to load history.")
        else:
            try:
                rows = supabase_fetch_lifts(supabase, history_user, limit=limit)
                if not rows:
                    st.info("No lifts found.")
                else:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not fetch lifts: {exc}")


def render_qa_tab(
    qdrant: QdrantClient | None, gemini_ready: bool, supabase: Client | None
) -> None:
    st.subheader("Ask Dr Opak (Qdrant + Gemini)")
    if not gemini_ready:
        st.warning("Connect Gemini first (`GEMINI_API_KEY`).")
        return

    use_qdrant_retrieval = st.checkbox(
        "Use Qdrant retrieval",
        value=True,
        help="Uses local sentence-transformer embeddings to query your Qdrant corpus.",
    )

    collection_name = st.text_input(
        "Qdrant Collection Name",
        value=_env("QDRANT_COLLECTION") or _env("COLLECTION_NAME", "fitness_knowledge"),
        help="Collection should contain vectors and payload fields like `text`, `source`, `topic`.",
        disabled=not use_qdrant_retrieval,
    )
    topic_filter = st.text_input(
        "Optional Topic Filter",
        placeholder="nutrition / strength / hypertrophy",
        disabled=not use_qdrant_retrieval,
    )
    coach_user_id = st.text_input(
        "Athlete ID (optional — recent lifts from Supabase)",
        placeholder="Same User ID as in Lift Tracker",
        help="If set and Supabase is connected, Dr Opak sees your latest logged lifts when answering.",
    )
    coach_lift_limit = st.slider(
        "Lifts to include for coach context",
        min_value=5,
        max_value=40,
        value=15,
        help="Most recent first. Only used when Athlete ID is filled in.",
    )
    question = st.text_area("Ask a fitness question", placeholder="How do I improve my squat depth?")
    top_k = st.slider(
        "Unique papers to retrieve",
        min_value=1,
        max_value=12,
        value=6,
        disabled=not use_qdrant_retrieval,
        help="After quality/year filters, duplicates from the same paper are merged.",
    )
    quality_min = st.slider(
        "Min quality score",
        min_value=0,
        max_value=100,
        value=60,
        disabled=not use_qdrant_retrieval,
    )
    year_from = st.slider(
        "Papers from year ≥",
        min_value=1990,
        max_value=2030,
        value=2010,
        disabled=not use_qdrant_retrieval,
    )

    if st.button("Answer Question"):
        if not question.strip():
            st.info("Type a question first.")
            return
        if use_qdrant_retrieval and not qdrant:
            st.error("Qdrant retrieval is enabled, but Qdrant is not configured.")
            return
        if use_qdrant_retrieval and not collection_name:
            st.error("Collection name is required.")
            return

        coach_uid = (coach_user_id or "").strip()
        lift_context: str | None = None
        if coach_uid:
            if not supabase:
                st.warning("Supabase is not connected; Dr Opak cannot load lift history for that ID.")
            else:
                try:
                    lift_rows = supabase_fetch_lifts(supabase, coach_uid, limit=int(coach_lift_limit))
                    lift_context = format_lifts_for_llm(lift_rows, coach_uid)
                except Exception as exc:
                    st.warning(f"Could not load lifts for coach: {exc}")

        try:
            if use_qdrant_retrieval:
                with st.spinner("Searching Qdrant and generating answer..."):
                    context_chunks = qdrant_search_context(
                        qdrant=qdrant,
                        collection_name=collection_name,
                        question=question,
                        top_k=top_k,
                        focus_topic=topic_filter.strip() or None,
                        quality_min=quality_min,
                        year_from=year_from,
                    )
                    if not context_chunks:
                        st.warning(
                            "No papers matched your filters after search. "
                            "Try lowering **Min quality score** or **Papers from year**."
                        )
                        return
                    answer = answer_with_context(question, context_chunks, lift_context)
            else:
                with st.spinner("Generating answer with Gemini..."):
                    context_chunks = []
                    answer = answer_without_context(question, lift_context)

            st.markdown("### Answer")
            if use_qdrant_retrieval:
                body = postprocess_rag_answer_body(answer)
                st.markdown(ensure_inline_source_hint(body))
            else:
                st.markdown(answer)

            if use_qdrant_retrieval:
                st.markdown("### References")
                st.caption("Numbered to match [Source N] in the answer above.")
                st.markdown(build_sources_markdown(context_chunks))
                with st.expander("Retrieved Context"):
                    if not context_chunks:
                        st.caption("No context retrieved.")
                    for idx, chunk in enumerate(context_chunks, start=1):
                        st.markdown(
                            f"**{idx}.** score={chunk['score']:.4f} | source={chunk['source']} | topic={chunk['topic']}"
                        )
                        if chunk["title"] or chunk["year"]:
                            st.caption(f"{chunk['title']} ({chunk['year']})")
                        if chunk["doi"]:
                            st.markdown(f"DOI: [{chunk['doi']}]({chunk['doi']})")
                        elif chunk["pubmed_url"]:
                            st.markdown(f"URL: [{chunk['pubmed_url']}]({chunk['pubmed_url']})")
                        st.write(chunk["text"])
                        st.markdown("---")

            if coach_uid and lift_context:
                with st.expander("Lift context sent to Dr Opak"):
                    st.text(lift_context)
        except Exception as exc:
            st.error(f"Could not answer question: {exc}")


def main() -> None:
    render_dr_opak_banner()
    st.caption(
        "Track lifts with Supabase · Ask Dr Opak with Qdrant + Gemini · "
        "*Not a real physician—definitely a real fan of progressive overload.*"
    )

    render_env_status()

    supabase = get_supabase_client()
    qdrant = get_qdrant_client()
    gemini_ready = init_gemini()

    tab_lifts, tab_qa = st.tabs(["Lift Tracker", "Ask Dr Opak"])
    with tab_lifts:
        render_lifts_tab(supabase)
    with tab_qa:
        render_qa_tab(qdrant, gemini_ready, supabase)

    st.divider()
    st.caption(
        "Dr Opak Industries — *A spotter is just a hug from science.* "
        "If Qdrant shows 404, confirm your cluster URL in the Qdrant Cloud console (HTTPS + port 6333, no `/collections` suffix)."
    )


if __name__ == "__main__":
    main()
