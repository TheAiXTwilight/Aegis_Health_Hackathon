# Aegis Health — Knowledge Base Version


## Current Status

Knowledge base not yet built.

TriageReport.knowledge_base_version and TriageReport.knowledge_base_date
are currently null in all generated reports.

These will be populated at application startup once the corpus is built
and this file contains the required fields.


## Required Fields

Once the corpus is built, this file must contain:

    snapshot_date: YYYY-MM-DD
    source_url: https://medlineplus.gov/xml.html
    git_commit: <full sha>

The application reads these fields at startup and injects them
into every TriageReport.


## Build Instructions (Mac only — not on Jetson)

Run in order:

    python rag/download_corpus.py
    python rag/export_minilm_onnx.py
    python rag/build_chroma.py
    python rag/build_faiss.py

Then append version metadata:

    echo "snapshot_date: $(date -u +%Y-%m-%d)" >> docs/corpus_version.md
    echo "source_url: https://medlineplus.gov/xml.html" >> docs/corpus_version.md
    echo "git_commit: $(git rev-parse HEAD)" >> docs/corpus_version.md

Then commit the built assets:

    git add data/ docs/corpus_version.md
    git commit -m "build: add knowledge base v1.0"


## Committed Assets

Once built, the following are committed to the repository:

    data/knowledge/minilm.onnx      ONNX-exported MiniLM embeddings model
    data/knowledge/chroma/          ChromaDB vector store (primary retriever)
    data/knowledge/faiss.index      FAISS index (fallback retriever)
    data/knowledge/faiss.docs       FAISS document store

data/knowledge/raw/ is gitignored — raw downloaded corpus is not committed.


## Planned Knowledge Base Sources

Source                          Version     Status
MedlinePlus (NIH)               v1.0        Not yet built
PubMed Abstracts                v2.0        Planned
NIH Clinical Guidelines         v2.0        Planned
WHO Public Health Guidance      v2.0        Planned


## Retriever Fallback Chain

Primary:    ChromaDB with ONNX MiniLM embeddings
Fallback:   FAISS with ONNX MiniLM embeddings

Zero results is valid output: RAGSearchResult(passages=[], retrieval_successful=True)
Mechanism failure returns ToolError(fatal=False) — pipeline continues without RAG.