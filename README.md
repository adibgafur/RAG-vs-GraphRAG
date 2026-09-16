<div align="center">

# 🔬 RAG vs GraphRAG — A Practical Comparison

**Two fully working Retrieval-Augmented Generation systems built on the same transcript corpus, designed to demonstrate when and why a knowledge graph dramatically improves retrieval quality.**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Engine-blue)](https://networkx.org)

</div>

---

## 📋 Overview

This project implements and compares three RAG architectures of increasing sophistication:

| Feature | Simple RAG | GraphRAG | Adaptive GraphRAG |
|---------|-----------|----------|--------------------|
| **Retrieval** | Hybrid dense + BM25 | Hybrid + entity graph expansion | Multi-hop adaptive traversal |
| **Query Modes** | 1 | 3 (Local · Global · Hybrid) | Auto-classified (4 types) |
| **Entity Awareness** | None | spaCy NER + co-occurrence graph | Typed relations with provenance |
| **Cross-doc Synthesis** | Limited | Community detection + LLM summaries | Multi-hop reasoning paths |
| **Explainability** | None | Entity/community labels | Full reasoning path traces |
| **Evaluation** | — | RAGAS + graph metrics | RAGAS + graph metrics |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Input Transcript                             │
│              (SRT / Plain Text / Timestamped .txt)                  │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Shared Ingestion Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────────┐  │
│  │  Parse &  │→│  Clean & │→│  Chunk     │→│  Embed (BGE)    │  │
│  │  Detect   │  │  Dedupe  │  │  (500 tok) │  │  + BM25 Index   │  │
│  └──────────┘  └──────────┘  └────────────┘  └─────────────────┘  │
└─────────────────┬──────────────────────┬────────────────────────────┘
                  │                      │
        ┌─────────┘                      └──────────┐
        ▼                                           ▼
┌───────────────────┐                    ┌───────────────────────────┐
│   Simple RAG      │                    │      GraphRAG             │
│                   │                    │                           │
│  Dense + BM25     │                    │  ┌─────────────────────┐  │
│  → RRF Fusion     │                    │  │ spaCy NER           │  │
│  → Cross-Encoder  │                    │  │ → Co-occurrence     │  │
│  → LLM Generate   │                    │  │   Graph (NetworkX)  │  │
│                   │                    │  │ → Community Detect   │  │
└───────────────────┘                    │  │ → LLM Summaries     │  │
                                         │  └─────────────────────┘  │
                                         │                           │
                                         │  Query Modes:             │
                                         │  • Local  (graph expand)  │
                                         │  • Global (community)     │
                                         │  • Hybrid (both)          │
                                         └───────────────────────────┘
                                                    │
                                                    ▼
                                         ┌───────────────────────────┐
                                         │  Adaptive GraphRAG (src/) │
                                         │                           │
                                         │  • Query Classifier       │
                                         │  • Typed Relations        │
                                         │  • Multi-Hop Retriever    │
                                         │  • Reasoning Paths        │
                                         └───────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── Simple_RAG/
│   ├── rag_pipeline.py          # Chunking, embedding, BM25, RRF, reranking, generation
│   ├── app.py                   # Streamlit chat UI
│   ├── requirements.txt
│   └── README.md
│
├── GraphRAG/
│   ├── graph_pipeline.py        # NER, entity graph, communities, 3 query modes
│   ├── app.py                   # Streamlit chat + Knowledge Graph inspector
│   ├── requirements.txt
│   └── README.md
│
├── src/                         # Adaptive GraphRAG core library
│   ├── __init__.py
│   ├── adaptive_pipeline.py     # Main pipeline: ties everything together
│   ├── query_classifier.py      # Classify queries → factual/relational/thematic/multi_hop
│   ├── multi_hop_retriever.py   # 0–3 hop graph traversal with relevance decay
│   ├── relation_extractor.py    # Typed, weighted, provenance-tagged relations
│   └── reasoning_path.py        # Human-readable traversal explanations
│
├── tests/
│   └── test_adaptive_pipeline.py
│
├── evaluate.py                  # RAGAS evaluation engine (3 modes × 10 questions)
├── eval_app.py                  # Streamlit evaluation dashboard
├── eval_requirements.txt
├── EVALUATION_GUIDE.md          # Step-by-step evaluation walkthrough
├── run_demo.py                  # Quick-start demo script
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- An LLM API key: [OpenAI](https://platform.openai.com/api-keys) or [Groq](https://console.groq.com/keys)

### 1. Clone the Repository

```bash
git clone https://github.com/sjsoumil/RAG-vs-GraphRAG.git
cd RAG-vs-GraphRAG
```

### 2. Install Dependencies

```bash
# GraphRAG (recommended — includes everything)
cd GraphRAG
pip install -r requirements.txt
pip install spacy numpy sentence-transformers
python -m spacy download en_core_web_sm
```

### 3. Set Up Environment

Create a `.env` file in the project root:

```bash
# Choose one (or both):
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

### 4. Run the App

```bash
# From the project root
streamlit run GraphRAG/app.py
```

The app will be available at **http://localhost:8501**.

### 5. Try Simple RAG (for comparison)

```bash
cd Simple_RAG
pip install -r requirements.txt
streamlit run app.py
```

---

## 📝 Supported Input Formats

The ingestion pipeline **auto-detects** the file format:

| Format | Example | Detection |
|--------|---------|-----------|
| **SRT subtitles** | `00:00:01,000 --> 00:00:03,500` | Standard SRT parsing |
| **Timestamped text** | `[00:00:01] Hello world...` | Bracket-timestamp parsing |
| **Plain text** | `This is a transcript...` | Line-by-line fallback |

---

## 🔍 How the Three Systems Compare

### Simple RAG
Standard hybrid retrieval: dense embeddings (BGE) + BM25 sparse search, fused with Reciprocal Rank Fusion and reranked with a cross-encoder.

### GraphRAG
Extends Simple RAG with a **knowledge graph layer**:
1. **Entity Extraction** — spaCy NER identifies people, organizations, concepts
2. **Co-occurrence Graph** — Entities appearing in the same chunk are connected (NetworkX)
3. **Community Detection** — Greedy modularity finds topic clusters
4. **Community Summaries** — LLM generates thematic summaries per community

Three query modes:
- **Local** — Retrieves chunks + expands via 1-hop entity graph neighbors
- **Global** — Routes queries to semantically matching community summaries
- **Hybrid** — Combines both strategies for comprehensive answers

### Adaptive GraphRAG (`src/`)
The most advanced tier, adding:
- **Query Classification** — Auto-detects if a question is factual, relational, thematic, or multi-hop
- **Typed Relations** — Extracts semantic relationships (e.g., `WORKS_AT`, `FOUNDED`, `DISAGREES_WITH`) with weights and provenance
- **Multi-Hop Retrieval** — Traverses 0–3 hops with relevance decay, adapting depth to query complexity
- **Reasoning Paths** — Generates human-readable explanations of how the answer was derived

---

## 📊 Evaluation

The evaluation framework benchmarks GraphRAG's three modes against 10 hand-crafted questions:

| Category | Questions | Expected Winner |
|----------|-----------|-----------------|
| Factual (3) | Single-source specific facts | Local mode |
| Relational (4) | Entity connections across docs | Local mode |
| Thematic (3) | Synthesis across all episodes | Global mode |

### Metrics

**RAGAS (LLM-as-judge):**
- **Faithfulness** — Is the answer grounded in retrieved context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Of retrieved chunks, how many were relevant?
- **Context Recall** — Did retrieval surface all needed information?

**Graph-specific:**
- **Entity Coverage Rate** — % of ground-truth entities in retrieved context
- **Graph Utilization** — Fraction of chunks sourced from graph expansion
- **Community Coherence** — % of a community's entities in its LLM summary

### Run Evaluation

```bash
pip install -r eval_requirements.txt

# Run evaluation
python evaluate.py --provider openai --api_key sk-...
# or
python evaluate.py --provider groq --api_key gsk_...

# View interactive dashboard
streamlit run eval_app.py
```

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed instructions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Embeddings** | `BAAI/bge-small-en-v1.5` (local, 384-dim) |
| **Vector Store** | Numpy + Pickle (cosine similarity, persistent) |
| **Sparse Retrieval** | BM25Okapi (`rank-bm25`) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **NER** | spaCy `en_core_web_sm` |
| **Graph Engine** | NetworkX |
| **Community Detection** | Greedy modularity (NetworkX) |
| **LLM** | OpenAI `gpt-4o-mini` · Groq `llama-3.1-70b` |
| **Evaluation** | RAGAS |
| **UI** | Streamlit |

---

## 💡 When to Use Each System

| Scenario | Recommended System |
|----------|-------------------|
| Straightforward factual Q&A | Simple RAG |
| Fast ingest, low latency needed | Simple RAG |
| Small, homogeneous corpus | Simple RAG |
| Questions about relationships between entities | GraphRAG |
| Thematic synthesis across documents | GraphRAG (Global mode) |
| Need to inspect *why* content was retrieved | GraphRAG |
| Complex multi-hop reasoning | Adaptive GraphRAG |
| Need transparent reasoning paths | Adaptive GraphRAG |

---

## 💬 Sample Questions to Try

**Factual** (Simple RAG excels)
- "What is Andrew Huberman's recommendation for falling asleep?"
- "What does Huberman say about morning sunlight exposure?"

**Relational** (GraphRAG Local excels)
- "How does Huberman connect dopamine to motivation and habits?"
- "What relationship does Huberman describe between sleep and cognitive performance?"

**Thematic** (GraphRAG Global excels)
- "What are the main themes Huberman discusses about brain optimization?"
- "How does Huberman's advice on stress management relate to his views on exercise?"

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ to demonstrate the power of knowledge graphs in retrieval-augmented generation.**

</div>
