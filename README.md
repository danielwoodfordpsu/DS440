# Fitness Tracker + AI Coach (Streamlit)

Two-tab Streamlit app:
- **Lift Tracker tab**: stores and reads user lift logs from Supabase.
- **AI Fitness Q&A tab**: retrieves context from Qdrant Cloud and answers with Gemini.

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure environment

```bash
cp .env.example .env
```

Fill in:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION` (optional, defaults to `fitness_knowledge`)
- `GEMINI_API_KEY`

## 3) Supabase table setup

Run this SQL in Supabase SQL editor:

```sql
create extension if not exists "pgcrypto";

create table if not exists public.lifts (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  exercise text not null,
  weight numeric not null,
  reps int not null,
  sets int not null,
  notes text,
  lifted_at timestamp without time zone not null,
  created_at timestamp without time zone not null default now()
);
```

If RLS is enabled, create policies that allow your app user to insert/select as needed.

## 4) Qdrant collection expectation

Your `QDRANT_COLLECTION` should contain points with:
- vector embeddings (same dimension as `sentence-transformers/all-MiniLM-L6-v2` if using default retrieval mode)
- payload fields:
  - `text` (chunk content)
  - `source` (optional source label)
  - `topic` (optional topic label used for filtering)

## 5) Run app

```bash
streamlit run app.py
```

## Notes

- Default retrieval uses local sentence-transformer embeddings (`all-MiniLM-L6-v2`).
- The default answer model is `models/gemini-2.5-flash`.
- The sidebar shows whether required environment variables are loaded.
