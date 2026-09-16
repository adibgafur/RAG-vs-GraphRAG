"""
Adaptive Multi-Hop GraphRAG Pipeline — Phase 5
================================================
Main pipeline unifying:
  - RelationExtractor   : Typed, weighted, provenance-tagged relationships (MultiDiGraph)
  - QueryClassifier     : Dynamic classification (factual, relational, thematic, multi_hop)
  - MultiHopRetriever   : Adaptive 0–3 hop graph traversal with relevance decay
  - ReasoningPath       : Step-by-step traversal explanations with supporting evidence

Combines hybrid vector/BM25 retrieval, typed knowledge graphs, modular community detection,
and adaptive multi-hop traversal into a single unified pipeline.
"""

import time
import os
import sys
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Set, Any
import numpy as np

import networkx as nx

# Add workspace root to path to ensure GraphRAG imports work if needed
_workspace_root = Path(__file__).resolve().parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from src.query_classifier import QueryClassifier, QueryClassification, classification_badge_html
from src.relation_extractor import RelationExtractor, Relation, merge_relations
from src.multi_hop_retriever import MultiHopRetriever, TraversalLog, TraversalResult, suggest_hops
from src.reasoning_path import (
    ReasoningPath,
    ReasoningStep,
    build_reasoning_path,
    format_reasoning_path,
    format_reasoning_path_html,
)

# Attempt importing GraphRAG modules if present
try:
    from GraphRAG.graph_pipeline import (
        TextChunk,
        parse_srt,
        clean_srt_blocks,
        reconstruct_full_text,
        chunk_segments,
        VectorStore,
        BM25Index,
        Reranker,
        reciprocal_rank_fusion,
    )
    _HAS_GRAPHRAG_BASE = True
except ImportError:
    _HAS_GRAPHRAG_BASE = False


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class QueryLog:
    """Telemetry and diagnostics log for a single query execution."""
    query: str
    classification: QueryClassification
    retrieval_mode: str
    max_hops_allowed: int
    actual_hops_used: int
    latency_ms: float
    reasoning_path: ReasoningPath
    retrieved_chunks_count: int
    entities_involved: List[str]
    pruned_paths: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "classification": {
                "type": self.classification.query_type,
                "confidence": round(self.classification.confidence, 3),
                "suggested_hops": self.classification.suggested_hops,
                "reasoning": self.classification.reasoning,
                "method": self.classification.method,
            },
            "retrieval_mode": self.retrieval_mode,
            "max_hops_allowed": self.max_hops_allowed,
            "actual_hops_used": self.actual_hops_used,
            "latency_ms": round(self.latency_ms, 2),
            "retrieved_chunks_count": self.retrieved_chunks_count,
            "entities_involved_count": len(self.entities_involved),
            "pruned_paths": self.pruned_paths,
        }


# ─── Fallback TextChunk when GraphRAG.graph_pipeline is not imported ────────

if not _HAS_GRAPHRAG_BASE:
    @dataclass
    class TextChunk:
        chunk_id: int
        text: str
        start_time: str
        end_time: str
        char_start: int
        char_end: int
        source: str
        token_count: int = 0


# ─── Adaptive GraphRAG Pipeline ─────────────────────────────────────────────

class AdaptiveGraphRAG:
    """
    Adaptive Multi-Hop Knowledge Graph RAG Pipeline.

    Parameters
    ----------
    persist_dir : str
        Directory to persist vector DB, graph store, and cached metadata.
    llm_fn : callable, optional
        Function(prompt: str) -> str used for relation extraction, query classification,
        and community summaries.
    spacy_model : str
        spaCy model for entity recognition (default "en_core_web_sm").
    decay_factor : float
        Relevance decay per hop during multi-hop graph traversal (default 0.7).
    relevance_threshold : float
        Minimum score required to expand a graph node (default 0.3).
    top_k_retrieve : int
        Number of candidates to retrieve in initial dense/sparse search (default 20).
    top_n_rerank : int
        Final top passages to return after reranking (default 5).
    """

    def __init__(
        self,
        persist_dir: str = "./adaptive_graph_store",
        llm_fn: Optional[Callable[[str], str]] = None,
        spacy_model: str = "en_core_web_sm",
        decay_factor: float = 0.7,
        relevance_threshold: float = 0.3,
        top_k_retrieve: int = 20,
        top_n_rerank: int = 5,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.llm_fn = llm_fn
        self.spacy_model = spacy_model
        self.decay_factor = decay_factor
        self.relevance_threshold = relevance_threshold
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank = top_n_rerank

        # Core Modules
        self.relation_extractor = RelationExtractor(llm_fn=llm_fn, spacy_model=spacy_model)
        self.query_classifier = QueryClassifier(llm_fn=llm_fn)

        # Knowledge Graph (Typed MultiDiGraph)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entity_to_chunks: Dict[str, Set[str]] = {}

        # Vector Store & Search Index
        if _HAS_GRAPHRAG_BASE:
            try:
                self.vector_store = VectorStore(
                    persist_dir=str(self.persist_dir / "chroma"),
                    collection_name="adaptive_graphrag",
                )
                self.bm25_index = BM25Index()
                self.reranker = Reranker()
            except Exception:
                self.vector_store = None
                self.bm25_index = None
                self.reranker = None
        else:
            self.vector_store = None
            self.bm25_index = None
            self.reranker = None

        # Communities & State
        self._communities: Dict[int, dict] = {}
        self._community_summary_embeddings: Optional[np.ndarray] = None
        self._community_ids_ordered: List[int] = []
        self._all_chunks: List[TextChunk] = []
        self._ingested_sources: Dict[str, dict] = {}

        # Multi-Hop Retriever instance (re-initialized when graph updates)
        self.multi_hop_retriever = MultiHopRetriever(
            graph=self.graph,
            entity_to_chunks=self._entity_to_chunks,
            decay_factor=self.decay_factor,
            relevance_threshold=self.relevance_threshold,
        )

        # Attempt to load state if exists
        self.load_state()

    def _update_retriever(self):
        """Update MultiHopRetriever with the latest graph and chunk map."""
        self.multi_hop_retriever = MultiHopRetriever(
            graph=self.graph,
            entity_to_chunks=self._entity_to_chunks,
            decay_factor=self.decay_factor,
            relevance_threshold=self.relevance_threshold,
        )

    # ── Ingestion & Typed Graph Construction ──────────────────────────────────

    def ingest_chunks(self, chunks: List[TextChunk], source_name: str = "corpus") -> dict:
        """
        Ingest a list of TextChunk objects:
          1. Extract typed relations via RelationExtractor
          2. Build/update typed MultiDiGraph
          3. Index vectors and BM25
          4. Detect communities & update summaries
        """
        if not chunks:
            return {"source": source_name, "chunks": 0}

        print(f"[INGEST] Ingesting {len(chunks)} chunks for source '{source_name}'...")

        # Add chunks to internal store
        self._all_chunks.extend(chunks)

        # Build Vector DB & BM25
        if self.vector_store:
            self.vector_store.index_chunks(chunks, self._hash(source_name))
            self.bm25_index.build(self._all_chunks)

        # Extract typed relations and build MultiDiGraph
        print("[GRAPH] Extracting typed relations and populating MultiDiGraph...")
        raw_relations = self.relation_extractor.extract_batch(chunks, source_name=source_name)
        merged_rels = merge_relations(raw_relations)

        for rel in merged_rels:
            # Ensure nodes exist
            for node in (rel.source, rel.target):
                if not self.graph.has_node(node):
                    self.graph.add_node(node, count=0)
                self.graph.nodes[node]["count"] = self.graph.nodes[node].get("count", 0) + 1

                if node not in self._entity_to_chunks:
                    self._entity_to_chunks[node] = set()
                self._entity_to_chunks[node].add(rel.source_chunk_id)

            # Add typed directed edge
            self.graph.add_edge(
                rel.source,
                rel.target,
                relation_type=rel.relation_type,
                weight=rel.weight,
                confidence=rel.confidence,
                source_doc_id=rel.source_doc_id,
                source_chunk_id=rel.source_chunk_id,
                raw_text=rel.raw_text,
            )

        # Also register all spaCy entities into node/chunk map even if no relation found
        for chunk in chunks:
            chunk_key = f"{getattr(chunk, 'source', source_name)}_{getattr(chunk, 'chunk_id', 0)}"
            ents = self.relation_extractor.extract_entities(chunk.text)
            for ent in ents:
                if not self.graph.has_node(ent):
                    self.graph.add_node(ent, count=0)
                self.graph.nodes[ent]["count"] = self.graph.nodes[ent].get("count", 0) + 1
                if ent not in self._entity_to_chunks:
                    self._entity_to_chunks[ent] = set()
                self._entity_to_chunks[ent].add(chunk_key)

        # Update communities & retriever
        self._detect_communities()
        if self.llm_fn:
            self.generate_summaries(self.llm_fn)

        self._update_retriever()
        self.save_state()

        stats = {
            "source": source_name,
            "chunks": len(chunks),
            "relations_extracted": len(merged_rels),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(self._communities),
        }
        self._ingested_sources[source_name] = stats
        return stats

    def ingest(self, file_path: str, llm_fn: Optional[Callable] = None) -> dict:
        """Parse .srt or .txt file and ingest chunks into AdaptiveGraphRAG."""
        if llm_fn:
            self.llm_fn = llm_fn
            self.relation_extractor.llm_fn = llm_fn
            self.query_classifier.llm_fn = llm_fn

        source_name = Path(file_path).stem
        if str(file_path).endswith(".srt"):
            return self.ingest_srt(file_path)

        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="replace")
        words = content.split()
        chunk_size_words, overlap_words = 300, 50
        chunks = []
        chunk_id = 0
        for i in range(0, len(words), chunk_size_words - overlap_words):
            chunk_words = words[i:i + chunk_size_words]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_time="00:00:00",
                end_time="00:00:00",
                char_start=0,
                char_end=len(chunk_text),
                source=source_name,
                token_count=len(chunk_words),
            ))
            chunk_id += 1
        return self.ingest_chunks(chunks, source_name=source_name)

    def ingest_srt(self, file_path: str) -> dict:
        """Parse SRT file and ingest chunks into AdaptiveGraphRAG."""
        if not _HAS_GRAPHRAG_BASE:
            raise NotImplementedError("GraphRAG base utilities required to parse SRT files.")

        source_name = Path(file_path).stem
        blocks = parse_srt(file_path)
        clean_b = clean_srt_blocks(blocks)
        segments = reconstruct_full_text(clean_b)
        chunks = chunk_segments(segments, source_name=source_name)
        return self.ingest_chunks(chunks, source_name=source_name)

    # ── Community Detection ───────────────────────────────────────────────────

    def _detect_communities(self):
        """
        Convert MultiDiGraph to weighted undirected graph and run modularity community detection.
        """
        if self.graph.number_of_nodes() == 0:
            return

        from networkx.algorithms.community import greedy_modularity_communities

        # Create collapsed undirected graph for community detection
        flat_graph = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            w = data.get("weight", 1.0)
            if flat_graph.has_edge(u, v):
                flat_graph[u][v]["weight"] += w
            else:
                flat_graph.add_edge(u, v, weight=w)

        for node in self.graph.nodes():
            if not flat_graph.has_node(node):
                flat_graph.add_node(node)

        all_communities = []
        for component in nx.connected_components(flat_graph):
            subg = flat_graph.subgraph(component)
            if subg.number_of_nodes() == 1:
                all_communities.append(frozenset(component))
            else:
                try:
                    comms = greedy_modularity_communities(subg, weight="weight")
                    all_communities.extend(comms)
                except Exception:
                    all_communities.append(frozenset(component))

        self._communities = {}
        for comm_id, members in enumerate(all_communities):
            members_sorted = sorted(
                members,
                key=lambda e: self.graph.nodes[e].get("count", 0),
                reverse=True,
            )
            chunk_keys: set = set()
            for ent in members_sorted:
                chunk_keys.update(self._entity_to_chunks.get(ent, set()))

            self._communities[comm_id] = {
                "entities": members_sorted,
                "chunk_keys": list(chunk_keys),
                "summary": None,
            }

        # Sort by size
        self._communities = dict(
            sorted(self._communities.items(), key=lambda x: -len(x[1]["entities"]))
        )
        self._communities = {i: v for i, v in enumerate(self._communities.values())}

    def generate_summaries(self, llm_fn: Callable[[str], str], force: bool = False):
        """Generate LLM community summaries."""
        chunk_lookup = self._make_chunk_lookup()
        generated = 0

        for comm_id, comm_data in self._communities.items():
            if comm_data["summary"] and not force:
                continue
            if len(comm_data["entities"]) < 2:
                comm_data["summary"] = f"Single entity cluster: {comm_data['entities'][0] if comm_data['entities'] else '—'}"
                continue

            top_entities = comm_data["entities"][:15]
            sample_keys = comm_data["chunk_keys"][:6]
            sample_texts = [chunk_lookup[k].text for k in sample_keys if k in chunk_lookup]
            context = "\n\n".join(sample_texts[:5]) if sample_texts else "(no text available)"

            prompt = (
                f"You are analyzing an entity cluster from a knowledge graph.\n"
                f"Key entities: {', '.join(top_entities)}\n\n"
                f"Excerpts:\n{context}\n\n"
                f"Write a 2-3 sentence summary of the main topic/theme this cluster covers."
            )
            try:
                summary = llm_fn(prompt)
            except Exception as e:
                summary = f"Cluster covering {', '.join(top_entities[:5])} ({e})"

            comm_data["summary"] = summary
            generated += 1

        self._cache_summary_embeddings()

    def _cache_summary_embeddings(self):
        """Cache community summary embeddings for global search."""
        if not self.vector_store:
            return
        summaries = [
            (cid, cd["summary"])
            for cid, cd in self._communities.items()
            if cd.get("summary") and len(cd["entities"]) >= 2
        ]
        if not summaries:
            self._community_summary_embeddings = None
            self._community_ids_ordered = []
            return

        ids = [s[0] for s in summaries]
        texts = [s[1] for s in summaries]
        embs = self.vector_store.embed_texts(texts)

        self._community_ids_ordered = ids
        self._community_summary_embeddings = np.array(embs)

    # ── Query API ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        mode: str = "auto",
        max_hops: Optional[int] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> dict:
        """
        Query the Adaptive Multi-Hop GraphRAG system.

        Parameters
        ----------
        question : str
            The user question.
        mode : str
            Query mode: "auto" (default, uses QueryClassifier), "local", "global", or "hybrid".
        max_hops : int, optional
            Override traversal depth (0–3). If None, calculated dynamically.
        llm_fn : callable, optional
            Override LLM function for answer generation.

        Returns
        -------
        dict containing answer, context, reasoning_path, reasoning_path_html, query_log, and hits.
        """
        start_time = time.time()

        # 1. Classification & Routing
        classification = self.query_classifier.classify(question)
        chosen_mode = classification.retrieval_mode if mode == "auto" else mode

        if max_hops is None:
            hops_to_use = classification.suggested_hops
        else:
            hops_to_use = max(0, min(3, max_hops))

        # 2. Extract Seed Entities
        query_ents = set(self.relation_extractor.extract_entities(question))

        # Vector / BM25 initial search for seeds and passages
        fused_hits: List[Tuple] = []
        if self.vector_store and self.bm25_index:
            dense = self.vector_store.dense_search(question, top_k=self.top_k_retrieve)
            sparse = self.bm25_index.search(question, top_k=self.top_k_retrieve)
            fused_hits = reciprocal_rank_fusion(dense, sparse)

        # Collect entities from top retrieved chunks
        chunk_ents: set = set()
        for text, _, _ in fused_hits[:5]:
            chunk_ents.update(self.relation_extractor.extract_entities(text))

        seed_entities = sorted(list(query_ents | chunk_ents))

        # 3. Traversal / Retrieval Routing
        chunk_lookup = self._make_chunk_lookup()
        traversal_result: Optional[TraversalResult] = None
        final_hits: List[Tuple] = []
        communities_used: List[int] = []

        if chosen_mode in ("local", "hybrid") or classification.query_type in ("factual", "relational", "multi_hop"):
            # Run multi-hop retriever
            traversal_result = self.multi_hop_retriever.retrieve(
                seed_entities=seed_entities,
                max_hops=hops_to_use,
                query=question,
            )

            # Convert traversal chunk_keys to hits
            graph_hits = []
            for key in traversal_result.chunk_keys:
                c = chunk_lookup.get(key)
                if c:
                    graph_hits.append((
                        c.text,
                        {
                            "source": c.source,
                            "chunk_id": c.chunk_id,
                            "start_time": c.start_time,
                            "end_time": c.end_time,
                            "token_count": c.token_count,
                        },
                        0.5,
                    ))

            combined = self._dedup(fused_hits + graph_hits)
            if self.reranker and combined:
                final_hits = self.reranker.rerank(question, combined[:25], top_n=self.top_n_rerank)
            else:
                final_hits = combined[:self.top_n_rerank]

        if chosen_mode in ("global", "hybrid") or classification.query_type == "thematic":
            # Global community retrieval
            global_res = self._global_community_search(question)
            communities_used = global_res["community_ids"]
            if chosen_mode == "global":
                final_hits = global_res["hits"]
            elif chosen_mode == "hybrid":
                # Merge local and global
                seen_keys = {f"{m.get('source')}_{m.get('chunk_id')}" for _, m, _ in final_hits}
                for t, m, s in global_res["hits"]:
                    k = f"{m.get('source')}_{m.get('chunk_id')}"
                    if k not in seen_keys:
                        final_hits.append((t, m, s * 0.8))
                if self.reranker and final_hits:
                    final_hits = self.reranker.rerank(question, final_hits[:25], top_n=self.top_n_rerank)

        # Ensure we have hits fallback if empty
        if not final_hits and fused_hits:
            final_hits = fused_hits[:self.top_n_rerank]

        # 4. Construct Reasoning Path
        traversal_paths = traversal_result.traversal_paths if traversal_result else None
        expanded_ents = sorted(list(traversal_result.expanded_entities)) if traversal_result else seed_entities
        all_chunk_ids = [f"{m.get('source')}_{m.get('chunk_id')}" for _, m, _ in final_hits]

        reasoning_path_obj = build_reasoning_path(
            query=question,
            query_type=classification.query_type,
            graph=self.graph,
            seed_entities=seed_entities,
            traversed_entities=expanded_ents,
            traversal_paths=traversal_paths,
            chunk_ids=all_chunk_ids,
            entity_to_chunks=self._entity_to_chunks,
            communities_used=communities_used,
        )

        # 5. Format Context & Generation
        formatted_context = self.format_context(
            hits=final_hits,
            summaries=[self._communities[cid]["summary"] for cid in communities_used if cid in self._communities],
            reasoning_path=reasoning_path_obj,
        )

        generator_fn = llm_fn or self.llm_fn
        answer = ""
        if generator_fn:
            generation_prompt = (
                f"You are an AI assistant answering questions based on knowledge graph retrieval.\n\n"
                f"CONTEXT:\n{formatted_context}\n\n"
                f"QUESTION: {question}\n\n"
                f"Answer the question clearly and directly based on the context provided."
            )
            try:
                answer = generator_fn(generation_prompt)
            except Exception as e:
                answer = f"Error generating answer: {e}"

        elapsed_ms = (time.time() - start_time) * 1000

        # Build Telemetry Log
        query_log = QueryLog(
            query=question,
            classification=classification,
            retrieval_mode=chosen_mode,
            max_hops_allowed=hops_to_use,
            actual_hops_used=traversal_result.log.actual_hops_used if traversal_result else 0,
            latency_ms=elapsed_ms,
            reasoning_path=reasoning_path_obj,
            retrieved_chunks_count=len(final_hits),
            entities_involved=expanded_ents,
            pruned_paths=traversal_result.log.pruned_paths if traversal_result else 0,
        )

        return {
            "question": question,
            "answer": answer,
            "context": formatted_context,
            "hits": final_hits,
            "classification": classification,
            "classification_badge_html": classification_badge_html(classification),
            "reasoning_path": reasoning_path_obj,
            "reasoning_path_text": format_reasoning_path(reasoning_path_obj),
            "reasoning_path_html": format_reasoning_path_html(reasoning_path_obj),
            "query_log": query_log,
            "entities": expanded_ents,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _global_community_search(self, question: str) -> dict:
        """Perform semantic search against community summaries."""
        if self._community_summary_embeddings is None or not self._community_ids_ordered:
            return {"hits": [], "community_ids": []}

        if not self.vector_store:
            return {"hits": [], "community_ids": []}

        query_emb = np.array(self.vector_store.embed_query(question))
        sims = self._community_summary_embeddings @ query_emb
        top_idx = np.argsort(sims)[::-1][:3]

        chunk_lookup = self._make_chunk_lookup()
        hits = []
        cids = []

        for idx in top_idx:
            cid = self._community_ids_ordered[idx]
            cd = self._communities.get(cid, {})
            if not cd:
                continue
            cids.append(cid)
            for key in cd["chunk_keys"][:6]:
                c = chunk_lookup.get(key)
                if c:
                    hits.append((
                        c.text,
                        {
                            "source": c.source,
                            "chunk_id": c.chunk_id,
                            "start_time": c.start_time,
                            "end_time": c.end_time,
                            "token_count": c.token_count,
                        },
                        float(sims[idx]),
                    ))

        return {"hits": self._dedup(hits), "community_ids": cids}

    def format_context(
        self,
        hits: List[Tuple],
        summaries: List[str],
        reasoning_path: ReasoningPath,
    ) -> str:
        """Format final context payload for LLM answer generation."""
        parts = []

        # Summaries
        if summaries:
            parts.append(
                "## Global Knowledge Graph Community Summaries\n" +
                "\n\n".join(f"• {s}" for s in summaries if s)
            )

        # Reasoning steps
        if reasoning_path and reasoning_path.steps:
            steps_str = "\n".join(
                f"• {s.source_entity} -> [{s.relation}] -> {s.target_entity} (confidence: {s.confidence:.2f})"
                for s in reasoning_path.steps[:8]
            )
            parts.append(f"## Multi-Hop Graph Traversal Chain\n{steps_str}")

        # Passage Excerpts
        if hits:
            excerpts = []
            for i, (text, meta, score) in enumerate(hits, 1):
                src = meta.get("source", "unknown")
                ts = f"{meta.get('start_time', '?')} -> {meta.get('end_time', '?')}"
                excerpts.append(f"[Excerpt {i} | Source: {src} | Timestamp: {ts}]\n{text}")
            parts.append("## Retrieved Passage Excerpts\n" + "\n\n---\n\n".join(excerpts))
        else:
            parts.append("No specific excerpts retrieved.")

        return "\n\n========================================\n\n".join(parts)

    def _make_chunk_lookup(self) -> Dict[str, TextChunk]:
        """Map 'source_chunkid' key -> TextChunk object."""
        lookup = {}
        for c in self._all_chunks:
            key = f"{c.source}_{c.chunk_id}"
            lookup[key] = c
        return lookup

    def _dedup(self, candidates: List[Tuple]) -> List[Tuple]:
        """Deduplicate candidates by (source, chunk_id)."""
        seen, deduped = set(), []
        for text, meta, score in candidates:
            key = f"{meta.get('source', '')}_{meta.get('chunk_id', '')}"
            if key not in seen:
                seen.add(key)
                deduped.append((text, meta, score))
        return deduped

    def _hash(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_state(self):
        """Save graph, entity map, communities, and chunks to disk."""
        state_file = self.persist_dir / "pipeline_state.pkl"
        try:
            with open(state_file, "wb") as f:
                pickle.dump(
                    {
                        "graph": self.graph,
                        "entity_to_chunks": self._entity_to_chunks,
                        "communities": self._communities,
                        "all_chunks": self._all_chunks,
                        "ingested_sources": self._ingested_sources,
                    },
                    f,
                )
            print(f"[SAVE] Saved pipeline state to {state_file}")
        except Exception as e:
            print(f"Failed to save state: {e}")

    def load_state(self):
        """Load pipeline state from disk if present."""
        state_file = self.persist_dir / "pipeline_state.pkl"
        if not state_file.exists():
            return
        try:
            with open(state_file, "rb") as f:
                data = pickle.load(f)
                self.graph = data.get("graph", nx.MultiDiGraph())
                self._entity_to_chunks = data.get("entity_to_chunks", {})
                self._communities = data.get("communities", {})
                self._all_chunks = data.get("all_chunks", [])
                self._ingested_sources = data.get("ingested_sources", {})

            self._update_retriever()
            if self._communities:
                self._cache_summary_embeddings()
            print(f"[LOAD] Loaded state from {state_file} ({self.graph.number_of_nodes()} nodes, {len(self._all_chunks)} chunks).")
        except Exception as e:
            print(f"Failed to load state: {e}")
