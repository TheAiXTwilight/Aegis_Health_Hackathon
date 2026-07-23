# Aegis Health — Knowledge Base Version

## Current Status

Knowledge base built — v0.2-medquad-partial.

TriageReport.knowledge_base_version / knowledge_base_date can now be
populated from the fields below.

## Build Summary (verified, not aspirational)

    corpus_kind:     medquad_partial
    source:          MedQuAD (github.com/abachaa/MedQuAD), CC BY 4.0
    source_subsets:  9 of 12 MedQuAD subsets (see below)
    source_documents: 1,500 QA-pair documents (proportionally sampled
                      across all 9 included subsets — see rag/medquad_source.py)
    n_chunks:        5,019
    chroma_vectors:  5,019 (verified via collection.count())
    faiss_vectors:   5,019 (verified via index.ntotal)
    snapshot_date:   2026-07-21
    git_commit:      not available — no .git in this build environment
                      (content hash / manual versioning until this repo
                      is under real git tracking)

Both counts match the corpus size exactly — no chunk loss during
build. Verified end-to-end with real (unmocked) queries through
tools/medical_rag_search.py: correct passages retrieved for chest
pain / angina and type 2 diabetes queries, and the ChromaDB→FAISS
fallback confirmed working by forcing a ChromaDB failure.

## Real limitations — read before treating this as complete

- **This is NOT the full MedQuAD corpus.** The full corpus (9 usable
  subsets) contains ~16,407 QA documents / ~53,000 chunks. Embedding
  that many chunks was measured at ~30-35 chunks/sec in this build
  environment (single CPU core), i.e. roughly 25-30 minutes — too
  long to complete within a single build step here, so this build
  used a 1,500-document sample. To build the full corpus, run:

      python rag/ingest.py --offline --medquad-max-documents 0

  (0 = no cap) somewhere with enough time budget for a ~25-30 min
  embedding step, or split the build into a scripted background job.

- **The 1,500-doc sample is proportionally stratified across all 9
  included subsets** (see rag/medquad_source.py — this was a real bug
  fix; an earlier version of this sampling took subsets sequentially
  and would have silently excluded MedlinePlus Health Topics, NIDDK,
  NINDS, NHLBI, SeniorHealth, and CDC entirely at this cap size).
  Coverage is still thin for topics outside the more heavily-weighted
  subsets (GARD genetic/rare disease and GHR genetics content make up
  ~66% of this sample) — expect weaker retrieval for common,
  non-genetic complaints not well covered by cardiac/diabetes/cancer
  examples tested so far.

- **Relevance threshold — IMPLEMENTED.** TOP_K=4 no longer always
  returns 4 results regardless of match quality. tools/medical_rag_search.py
  now filters by a calibrated distance/score cutoff before returning
  passages: ChromaDB (L2 distance, lower=better) rejects results with
  distance > 1.2; FAISS (inner-product score, higher=better) rejects
  results with score < 0.35. Both cutoffs were calibrated empirically
  against this build's real corpus and index — not guessed — using a
  spread of clearly in-corpus vs. clearly unrelated test queries (see
  git history / conversation log for the calibration data). A query
  with no good match now correctly returns
  RAGSearchResult(passages=[], retrieval_successful=True) instead of
  4 weak "least bad" matches presented as confident. Verified on both
  the ChromaDB primary path and the FAISS fallback path. If the
  corpus is rebuilt at a different size or composition, re-calibrate
  these two constants — see the comment above them in
  tools/medical_rag_search.py for the method.

- **3 MedQuAD subsets excluded entirely**: ADAM, MedlinePlus Drugs,
  MedlinePlus Herbs/Supplements — their <Answer> text ships stripped
  in the public MedQuAD release to respect MedlinePlus copyright (per
  MedQuAD's own README). rag/medquad_source.py skips these by design;
  they contribute zero documents regardless of cap size.

- **No unit tests written for rag/*.py build scripts themselves**
  (chunk.py, download_corpus.py, medquad_source.py, embed.py,
  build_chroma.py, build_faiss.py). Verified by running the pipeline
  directly and inspecting real output (citation uniqueness, vector
  counts, retrieval quality) — not by an automated test suite.
  tests/tools/test_medical_rag_search.py (18 tests, all passing,
  unchanged) covers the runtime consumer only, not the build package.

- **git_commit is not a real commit hash** — there's no git repository
  in this build environment. Treat the field above as a placeholder
  until this project is under real version control.

## Required Fields (reference — see Build Summary above for actual values)

    snapshot_date: YYYY-MM-DD
    source_url: https://medlineplus.gov/xml.html   (or MedQuAD repo URL if not live-sourced)
    git_commit: <full sha>

The application reads these fields at startup and injects them
into every TriageReport.

## Build Instructions

Requires a local MedQuAD checkout for the (recommended) offline-but-real
path, or network access to medlineplus.gov for the live path:

    git clone https://github.com/abachaa/MedQuAD data/knowledge/sources/medquad
    python rag/ingest.py --offline                         # capped default, see rag/ingest.py
    python rag/ingest.py --offline --medquad-max-documents 0   # full corpus, ~25-30 min

Or, for the live MedlinePlus source (used only if no local MedQuAD
checkout is present — see rag/download_corpus.py's fallback order):

    python rag/ingest.py

rag/export_minilm_onnx.py does NOT need to be re-run — this build
reused the already-exported data/knowledge/minilm.onnx and
tokenizer.json.

Then append real version metadata once this repo has git tracking:

    echo "snapshot_date: $(date -u +%Y-%m-%d)" >> docs/corpus_version.md
    echo "source_url: https://github.com/abachaa/MedQuAD" >> docs/corpus_version.md
    echo "git_commit: $(git rev-parse HEAD)" >> docs/corpus_version.md

Then commit the built assets:

    git add data/ docs/corpus_version.md
    git commit -m "build: knowledge base v0.2-medquad-partial (1,500 docs / 5,019 chunks)"

## Committed Assets

    data/knowledge/minilm.onnx      ONNX-exported MiniLM embeddings model (pre-existing, reused)
    data/knowledge/tokenizer.json   MiniLM tokenizer (pre-existing, reused)
    data/knowledge/chroma/          ChromaDB vector store (primary retriever) — 5,019 vectors
    data/knowledge/faiss.index      FAISS index (fallback retriever) — 5,019 vectors
    data/knowledge/faiss.docs       FAISS document store — 5,019 lines

data/knowledge/raw/ is gitignored — raw downloaded corpus is not committed.
data/knowledge/sources/medquad/ (the cloned MedQuAD repo) is also not
committed — it's a build-time dependency, not a build output; re-clone
it before re-running rag/ingest.py --offline.

## Actual Knowledge Base Sources (this build)

Subset                          Source                              Docs in this sample
1_CancerGov_QA                  NIH National Cancer Institute        67
2_GARD_QA                       NIH Genetic and Rare Diseases Center 493
3_GHR_QA                        NIH Genetics Home Reference          496
4_MPlus_Health_Topics_QA        MedlinePlus Health Topics            90
5_NIDDK_QA                      NIH NIDDK                            109
6_NINDS_QA                      NIH NINDS                            99
7_SeniorHealth_QA               NIH SeniorHealth                     70
8_NHLBI_QA_XML                  NIH NHLBI                            51
9_CDC_QA                        CDC                                  25
10/11/12 (ADAM/Drugs/Herbs)     excluded — answers stripped in MedQuAD's public release

## Planned Knowledge Base Sources (future work, not this build)

Source                          Version     Status
Full MedQuAD (all 9 subsets)    v0.3        Feasible, not yet run (time budget)
PubMed Abstracts                v2.0        Planned
NIH Clinical Guidelines         v2.0        Planned
WHO Public Health Guidance      v2.0        Planned

## Retriever Fallback Chain

Primary:    ChromaDB with ONNX MiniLM embeddings
Fallback:   FAISS with ONNX MiniLM embeddings

Zero results is valid output: RAGSearchResult(passages=[], retrieval_successful=True)
Mechanism failure returns ToolError(fatal=False) — pipeline continues without RAG.

Verified in this build: forced a ChromaDB query failure and confirmed
the pipeline fell through to FAISS and still returned correct,
relevant passages (see build log / manual verification above).

## Phase 2.5 Note — Conditional RAG

MedicalRAGSearch now runs conditionally based on plan.use_rag.

When plan.use_rag=True:
    MedicalRAGSearch runs normally.
    The retriever fallback chain applies as described above.
    Retrieved passages are included in the report Evidence section.

When plan.use_rag=False:
    MedicalRAGSearch is skipped entirely.
    state.rag_result stays None.
    The report Evidence section uses the standard "No evidence retrieved"
    message from ReportGenerator.
    RAG coverage in confidence formula reduces (RAG is always-submitted).

PlanValidator may force use_rag=True even when the planner said False,
if clinical safety signals are present (chest pain, troponin, critical
X-ray findings, or polypharmacy >3 medications). In this case
MedicalRAGSearch always runs regardless of the planner's preference.


## Retriever Fallback Chain

Primary:    ChromaDB with ONNX MiniLM embeddings
Fallback:   FAISS with ONNX MiniLM embeddings

Zero results is valid output: RAGSearchResult(passages=[], retrieval_successful=True)
Mechanism failure returns ToolError(fatal=False) — pipeline continues without RAG.


## Phase 2.5 Note — Conditional RAG

MedicalRAGSearch now runs conditionally based on plan.use_rag.

When plan.use_rag=True:
    MedicalRAGSearch runs normally.
    The retriever fallback chain applies as described above.
    Retrieved passages are included in the report Evidence section.

When plan.use_rag=False:
    MedicalRAGSearch is skipped entirely.
    state.rag_result stays None.
    The report Evidence section uses the standard "No evidence retrieved"
    message from ReportGenerator.
    RAG coverage in confidence formula reduces (RAG is always-submitted).

PlanValidator may force use_rag=True even when the planner said False,
if clinical safety signals are present (chest pain, troponin, critical
X-ray findings, or polypharmacy >3 medications). In this case
MedicalRAGSearch always runs regardless of the planner's preference.