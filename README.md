# Document Indexing & Semantic Search (Gemini + pgvector)

A small Python module that takes a PDF or DOCX file, extracts clean text, splits it into chunks
with one of three strategies, creates embeddings with Google's `gemini-embedding-001`, stores
everything in PostgreSQL with `pgvector`, and lets you search the content by meaning.

```
PDF/DOCX ──► extract & clean ──► chunk (fixed / sentence / paragraph) ──► Gemini embeddings ──► PostgreSQL + pgvector
                                                                                                    ▲
"login issue" ──► Gemini query embedding ──────────────── cosine similarity search ─────────────────┘
```

## Project structure

```
index_documents.py      CLI: index a document
search.py               CLI: semantic search
docsearch/
  config.py             loads GEMINI_API_KEY / POSTGRES_URL from .env
  extract.py            PDF (pypdf) and DOCX (python-docx) text extraction + cleaning
  chunking.py           fixed-size-with-overlap, sentence and paragraph strategies
  embeddings.py         Gemini embedding client (batching + retries)
  db.py                 table creation, inserts and vector similarity search
  errors.py             one exception type per handled failure
docs/                   sample documents (a fictional support knowledge base)
tests/                  offline tests (no API key or database needed)
```

## Installation

Requirements: Python 3.9+ and a PostgreSQL database with the `pgvector` extension available
(for example a free [Supabase](https://supabase.com) project, or a local `pgvector/pgvector` Docker image).

```bash
git clone <this-repo-url>
cd jeen-document-search
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env
```

## Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key from [Google AI Studio](https://aistudio.google.com) |
| `POSTGRES_URL` | PostgreSQL connection string, e.g. `postgresql://user:password@host:5432/postgres` |

Both are read from `.env` (which is git-ignored). Neither is ever printed.

The table is created automatically on the first indexing run:

| Column | Type | Notes |
|---|---|---|
| `id` | BIGSERIAL | primary key |
| `chunk_text` | TEXT | the chunk content |
| `embedding` | vector(768) | Gemini embedding |
| `filename` | TEXT | source file name |
| `split_strategy` | TEXT | `fixed`, `sentence` or `paragraph` |
| `chunk_index` | INTEGER | position of the chunk in the document |
| `created_at` | TIMESTAMPTZ | insertion time |

## Usage

### Index a document

```bash
python index_documents.py --file ./docs/support_knowledge_base.pdf --strategy paragraph
python index_documents.py --file ./docs/support_knowledge_base.docx --strategy sentence
python index_documents.py --file ./docs/support_knowledge_base.pdf --strategy fixed --chunk-size 800 --overlap 150
```

Re-indexing the same file with the same strategy replaces its previous chunks (no duplicates).

### Search

```bash
python search.py --query "login issue"
python search.py --query "how long does a refund take" --top-k 3
python search.py --query "account locked" --strategy sentence --min-score 0.6
```

Options: `--top-k` (default 5), `--strategy` and `--file` filters, `--min-score` similarity threshold (0–1).

## Sample output

### Indexing run

```
$ python index_documents.py --file ./docs/support_knowledge_base.pdf --strategy paragraph
[1/4] Extracted 4,776 characters from support_knowledge_base.pdf
[2/4] Split into 13 chunks using 'paragraph' strategy (avg 364 chars, min 203, max 744)
[3/4] Created 13 embeddings with gemini-embedding-001 (768 dimensions)
[4/4] Stored 13 chunks in PostgreSQL (table: document_chunks)
Done in 4.6s
```

### Search run

```
$ python search.py --query "login issue"
Top 5 results for "login issue":

#1  similarity=0.706  file=support_knowledge_base.pdf  strategy=paragraph  chunk=0  id=1
    Nimbus Cloud - Customer Support Knowledge Base 1. Login Issues Most login problems are caused by an
    incorrect password, an expired session, or a browser that blocks cookies. Before contacting support,
    customers should try signing in from a private browser window and make sure that cookies are enabled
    for nimbus.example.com.

#2  similarity=0.679  file=support_knowledge_base.pdf  strategy=paragraph  chunk=2  id=3
    Customers who use single sign-on (SSO) through their company should contact their own IT
    administrator first, because Nimbus Cloud does not manage company identity providers. Login issues
    affecting many users at the same company are usually caused by an expired SSO certificate.

#3  similarity=0.648  file=support_knowledge_base.pdf  strategy=paragraph  chunk=1  id=2
    If a customer enters a wrong password five times in a row, the account is locked for 30 minutes as a
    security measure. The lock is released automatically. Support agents can unlock the account earlier
    only after verifying the customer's identity with the email address on file.
...
```

A query with **no words in common** with the document still finds the right section — this is what semantic search adds over keyword search:

```
$ python search.py --query "I want my money back" --top-k 2 --strategy paragraph
Top 2 results for "I want my money back":

#1  similarity=0.652  file=support_knowledge_base.pdf  strategy=paragraph  chunk=6  id=7
    Refunds are available within 14 days of a charge for annual plans and within 48 hours for monthly
    plans. Refund requests are handled by the billing team and usually take five to seven business days
    to appear on the customer's statement. Partial refunds are not offered for unused days of a monthly
    plan.

#2  similarity=0.593  file=support_knowledge_base.pdf  strategy=paragraph  chunk=7  id=8
    Customers who see a charge they do not recognize should first check whether another administrator on
    the same workspace upgraded the plan. Duplicate charges are reversed automatically within three
    business days.
```

### Results stored in the database

The `document_chunks` table in PostgreSQL (Supabase SQL editor), showing the 13 paragraph chunks of
the sample PDF, each with a 768-dimensional embedding:

```sql
SELECT id, filename, split_strategy, chunk_index,
       left(chunk_text, 70) AS chunk_preview,
       vector_dims(embedding) AS dims,
       created_at
FROM document_chunks
WHERE split_strategy = 'paragraph'
ORDER BY chunk_index;
```

![document_chunks table](docs/supabase_table.png)

## Chunking strategies

| Strategy | How it works | Good for |
|---|---|---|
| `fixed` | Windows of ~800 characters, each overlapping the previous one by 150 characters; cuts happen at word boundaries | Unstructured text, predictable chunk sizes |
| `sentence` | Whole sentences grouped up to ~800 characters; a sentence is never cut | Dense text where each sentence carries a fact |
| `paragraph` | One chunk per paragraph; headings are attached to the paragraph after them; very long paragraphs fall back to sentence splitting | Well-structured documents (policies, FAQs, manuals) |

### Comparison on the sample document

The same PDF was indexed with all three strategies and queried with `"I want my money back"`:

| Strategy | Chunks | Top result | Similarity | Observation |
|---|---|---|---|---|
| `paragraph` | 13 (avg 364 chars) | The refund policy paragraph, exactly | **0.652** | One topic per chunk → the sharpest match |
| `sentence` | 7 (avg 678 chars) | End of refund policy + subscription plans | 0.606 | The refund policy was split across two chunks, each mixed with another topic |
| `fixed` | 8 (avg 723 chars) | Starts mid-sentence ("on file. If a payment fails…") | 0.600 | Contains the answer, but boundaries ignore meaning and the chunk mixes billing with plans |

**Takeaway:** chunks that hold a single topic produce sharper embeddings, so `paragraph` is the default
for structured documents like this one. `fixed` remains a safe fallback for text with no reliable
structure. Note that paragraph quality depends on the source format: DOCX keeps real paragraph
boundaries, while PDF only stores lines on a page, so paragraphs are reconstructed heuristically and
two paragraphs are occasionally merged (13 chunks from the PDF vs. 19 from the DOCX version of the same text).

## Error handling

| Situation | Behaviour |
|---|---|
| Missing file | `Error: File not found: <path>` |
| Unsupported / corrupted file | `Error: Unsupported file type '.txt'. Supported types: PDF, DOCX.` |
| No extractable text (e.g. scanned PDF) | `Error: No extractable text found ... it needs OCR first.` |
| Embedding failure | Retries rate-limit and server errors with backoff, then `Error: Gemini embedding request failed (HTTP 4xx/5xx)` |
| Database connection failure | Checked **before** calling Gemini so no API calls are wasted; the message never includes the connection string |
| Missing environment variables | Names the missing variable(s) only |
| Empty search results | `No results found for "<query>"` with a hint |

Run the offline tests with `python -m unittest discover tests`.

## Design decisions

- **768-dimensional embeddings.** `gemini-embedding-001` supports smaller outputs; 768 keeps storage small and allows a pgvector HNSW index (limited to 2,000 dimensions) with minimal quality loss.
- **Separate task types.** Chunks are embedded as `RETRIEVAL_DOCUMENT` and queries as `RETRIEVAL_QUERY`, which is how Gemini is optimised for search.
- **Cosine similarity.** Results are ranked by `embedding <=> query` (cosine distance) and shown as similarity (1 = identical meaning).
- **Atomic writes.** All embeddings are created before anything is written, and the delete + insert run in a single transaction, so a failure never leaves a half-indexed document.

## Deploying in private / air-gapped environments

The code is built so the two external dependencies can be swapped without touching the rest:

- **Database:** only a connection string is used, so any PostgreSQL with pgvector works — a managed service in the customer's own cloud (e.g. AWS RDS, Azure Database for PostgreSQL, Google Cloud SQL) or a self-hosted instance on-premises.
- **Embeddings:** all Gemini calls live in `docsearch/embeddings.py`. In an environment without internet access, this module can be replaced by a locally hosted open-source embedding model that returns vectors of the same size.
