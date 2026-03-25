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


load_dotenv()

st.set_page_config(page_title="Fitness Tracker + Coach", page_icon="🏋️", layout="wide")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_supabase_client() -> Client | None:
    url = _env("SUPABASE_URL")
    key = _env("SUPABASE_ANON_KEY") or _env("SUPABASE_PUBLISHABLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_qdrant_client() -> QdrantClient | None:
    url = _env("QDRANT_URL")
    api_key = _env("QDRANT_API_KEY")
    if not url or not api_key:
        return None
    return QdrantClient(url=url, api_key=api_key)


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


def answer_with_context(question: str, context_chunks: list[dict[str, Any]]) -> str:
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
You are a fitness science coach. Ground your answer in the retrieved papers below.

Evidence & citations:
- Use ONLY the retrieved context for factual claims (methods, findings, numbers, quotes). If something is not in the context, say so briefly or omit it.
- When you state a finding, add an inline citation: [Source N] or [Source N, Source M]. Put the citation at the end of the sentence or paragraph block it supports.
- Merge overlapping points so you are not repeating the same citation on many tiny one-liners; prefer **fewer, richer** points over many fragment bullets.

Depth & structure (aim for a helpful mini-brief, not a list of slogans):
- Write in **markdown**. Use a **short intro** (2–4 sentences) that frames the question and what the evidence addresses.
- Then use **subsections** with `###` headings (e.g. biomechanics, coaching, injury considerations) as fits the question.
- Under each theme, write **2–5 sentences** per point where useful: what was studied, **what they found**, and **what it implies practically** for the reader. Include specifics from the excerpts (e.g. population, task, direction of effect) when the text provides them.
- End with a **Practical takeaway** subsection: 3–5 concrete bullets tying advice together (still cited where evidence-based).

Do NOT add a "Sources" or "References" section — the app attaches links.
Do NOT print a top-level heading that says only "Answer" — you may use `###` for real section titles.

Question:
{question}

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


def answer_without_context(question: str) -> str:
    prompt = f"""
You are a helpful strength and fitness coach.
Give concise, safe, practical advice for the user's question.
If needed, mention uncertainty and suggest consulting a professional for injuries or medical concerns.

Question:
{question}
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text or "I could not generate an answer."


def render_env_status() -> None:
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
    st.subheader("Log Your Lifts")
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


def render_qa_tab(qdrant: QdrantClient | None, gemini_ready: bool) -> None:
    st.subheader("Fitness Q&A (Qdrant + Gemini)")
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
                    answer = answer_with_context(question, context_chunks)
            else:
                with st.spinner("Generating answer with Gemini..."):
                    context_chunks = []
                    answer = answer_without_context(question)

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
        except Exception as exc:
            st.error(f"Could not answer question: {exc}")


def main() -> None:
    st.title("🏋️ Fitness Tracker + AI Coach")
    st.caption("Track lifts with Supabase and ask fitness questions with Qdrant + Gemini.")

    render_env_status()

    supabase = get_supabase_client()
    qdrant = get_qdrant_client()
    gemini_ready = init_gemini()

    tab_lifts, tab_qa = st.tabs(["Lift Tracker", "AI Fitness Q&A"])
    with tab_lifts:
        render_lifts_tab(supabase)
    with tab_qa:
        render_qa_tab(qdrant, gemini_ready)


if __name__ == "__main__":
    main()
