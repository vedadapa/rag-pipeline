## What This Project Does

Takes 148,670 loan records, converts them to natural language, stores them as semantic vectors in ChromaDB, and uses Groq's LLaMA 3.1 to answer questions about defaulted customers based on retrieved context.

## Tech Stack

- **ChromaDB** — vector database for storing and querying embeddings
- **sentence-transformers** (`all-MiniLM-L6-v2`) — converts text to 384-dimensional semantic vectors
- **Groq** (`llama-3.1-8b-instant`) — LLM for generating answers from retrieved context
- **Pandas** — data preprocessing and text conversion

## Key Decisions & What I Learned

**Why sentence-transformers over ChromaDB's default embedding?**
ChromaDB's default embedding couldn't distinguish "defaulted" from "did not default" — the distances were identical (0.62 for both). Switching to `all-MiniLM-L6-v2` dropped distances to 0.31 and improved retrieval quality significantly.

**Why balanced sampling?**
The original dataset is heavily skewed — 112,031 non-defaulted vs 36,639 defaulted. Using `.head(500)` meant only 140 defaulted customers in ChromaDB, which hurt retrieval. Fixed by taking 250 from each class.

**Query wording matters in RAG**
"Customer with low credit score who defaulted" failed — the embedding model saw it as too similar to "did not default" sentences. "Defaulted loan customer" worked because it's direct and unambiguous. Lesson: retrieval quality depends on how you phrase the query, not just what model you use.

**Groq doesn't hallucinate**
When ChromaDB returned wrong documents, Groq honestly said "I don't see a defaulted customer in these records." That's the correct behavior — RAG quality is only as good as retrieval quality.

## How to Run

1. Add your Groq API key to Colab Secrets as `GROQ_API_KEY`
2. Add your Kaggle credentials for dataset download
3. Run `02_rag_pipeline.ipynb` top to bottom

## Dataset

[Loan Default Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
