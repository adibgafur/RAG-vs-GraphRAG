"""
Multi-Hop Retriever — Phase 3
===============================
Adaptive 0–3 hop graph traversal with relevance filtering at every hop
to avoid graph explosion.

The system decides traversal depth based on query complexity:
  - Factual queries  → 0-hop (vector-only)
  - Relational        → 1–2 hops
  - Complex synthesis → 2–3 hops

Score decay per hop prevents retrieval of irrelevant distant nodes.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TraversalLog:
    """Detailed log of a multi-hop traversal for evaluation."""
    query: str
    max_hops_allowed: int
    actual_hops_used: int
    nodes_traversed: int
    edges_traversed: int
    chunks_gathered: int
    latency_ms: float
    per_hop_stats: List[dict] = field(default_factory=list)
    pruned_paths: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "max_hops_allowed": self.max_hops_allowed,
            "actual_hops_used": self.actual_hops_used,
            "nodes_traversed": self.nodes_traversed,
            "edges_traversed": self.edges_traversed,
            "chunks_gathered": self.chunks_gathered,
            "latency_ms": round(self.latency_ms, 2),
            "per_hop_stats": self.per_hop_stats,
            "pruned_paths": self.pruned_paths,
        }


@dataclass
class TraversalResult:
    """Result of a multi-hop traversal."""
    expanded_entities: Set[str]
    traversal_paths: List[Tuple]     # (src, rel, tgt, score, hop_depth)
    chunk_keys: Set[str]
    log: TraversalLog


# ─── Multi-Hop Retriever ─────────────────────────────────────────────────────

class MultiHopRetriever:
    """
    Adaptive multi-hop graph traversal with relevance decay.

    At each hop, expands via typed edges, scoring:
      edge_weight × confidence × (decay_factor ^ hop_depth)

    Prunes paths below relevance_threshold to avoid explosion.

    Parameters
    ----------
    graph : nx.MultiDiGraph or nx.Graph
        The knowledge graph.
    entity_to_chunks : dict
        Entity → set of chunk keys.
    decay_factor : float
        Score multiplier per hop depth (default 0.7).
    relevance_threshold : float
        Minimum score to continue expanding (default 0.3).
    max_neighbors_per_hop : int
        Max neighbors to expand per entity per hop (default 8).
    """

    def __init__(
        self,
        graph,
        entity_to_chunks: Dict[str, set],
        decay_factor: float = 0.7,
        relevance_threshold: float = 0.3,
        max_neighbors_per_hop: int = 8,
    ):
        self.graph = graph
        self.entity_to_chunks = entity_to_chunks
        self.decay_factor = decay_factor
        self.relevance_threshold = relevance_threshold
        self.max_neighbors_per_hop = max_neighbors_per_hop

    def retrieve(
        self,
        seed_entities: List[str],
        max_hops: int = 2,
        query: str = "",
    ) -> TraversalResult:
        """
        Perform adaptive multi-hop traversal from seed entities.

        Parameters
        ----------
        seed_entities : list[str]
            Starting entities (from query + top retrieved chunks).
        max_hops : int
            Maximum traversal depth (0–3). 0 means no graph expansion.
        query : str
            Original query for logging.

        Returns
        -------
        TraversalResult with expanded entities, paths, chunk keys, and log.
        """
        import networkx as nx

        start_time = time.time()

        max_hops = max(0, min(3, max_hops))  # Clamp to [0, 3]

        all_entities: Set[str] = set()
        all_paths: List[Tuple] = []
        all_chunk_keys: Set[str] = set()
        edges_traversed = 0
        pruned = 0
        per_hop_stats = []

        # Add seed entity chunks
        for ent in seed_entities:
            if ent in self.entity_to_chunks:
                all_chunk_keys.update(self.entity_to_chunks[ent])
            all_entities.add(ent)

        if max_hops == 0:
            elapsed = (time.time() - start_time) * 1000
            log = TraversalLog(
                query=query, max_hops_allowed=0, actual_hops_used=0,
                nodes_traversed=len(all_entities), edges_traversed=0,
                chunks_gathered=len(all_chunk_keys), latency_ms=elapsed,
            )
            return TraversalResult(
                expanded_entities=all_entities,
                traversal_paths=[],
                chunk_keys=all_chunk_keys,
                log=log,
            )

        # Current frontier: entities to expand from, with their scores
        frontier: Dict[str, float] = {ent: 1.0 for ent in seed_entities if ent in self.graph}
        visited: Set[str] = set(seed_entities)
        actual_hops = 0

        is_multigraph = isinstance(self.graph, nx.MultiDiGraph)

        for hop in range(1, max_hops + 1):
            if not frontier:
                break

            hop_start = time.time()
            new_frontier: Dict[str, float] = {}
            hop_edges = 0
            hop_new_entities = 0
            hop_pruned = 0

            for src_entity, src_score in frontier.items():
                # Get neighbors
                if is_multigraph:
                    neighbors = self._get_multigraph_neighbors(src_entity)
                else:
                    neighbors = self._get_graph_neighbors(src_entity)

                # Sort by edge weight, take top N
                neighbors.sort(key=lambda x: x[2], reverse=True)
                neighbors = neighbors[:self.max_neighbors_per_hop]

                for tgt, rel_type, edge_weight in neighbors:
                    hop_edges += 1
                    edges_traversed += 1

                    # Score with decay
                    path_score = src_score * edge_weight * (self.decay_factor ** hop)

                    if path_score < self.relevance_threshold:
                        hop_pruned += 1
                        pruned += 1
                        continue

                    all_paths.append((src_entity, rel_type, tgt, round(path_score, 3), hop))

                    if tgt not in visited:
                        visited.add(tgt)
                        all_entities.add(tgt)
                        hop_new_entities += 1

                        # Collect chunks for this entity
                        if tgt in self.entity_to_chunks:
                            all_chunk_keys.update(self.entity_to_chunks[tgt])

                        # Only add to frontier if score is high enough to continue
                        if path_score > self.relevance_threshold * 1.5:
                            new_frontier[tgt] = max(
                                new_frontier.get(tgt, 0), path_score
                            )

            hop_elapsed = (time.time() - hop_start) * 1000
            per_hop_stats.append({
                "hop": hop,
                "frontier_size": len(frontier),
                "edges_examined": hop_edges,
                "new_entities": hop_new_entities,
                "pruned": hop_pruned,
                "latency_ms": round(hop_elapsed, 2),
            })

            if new_frontier:
                actual_hops = hop
            frontier = new_frontier

        elapsed = (time.time() - start_time) * 1000
        log = TraversalLog(
            query=query,
            max_hops_allowed=max_hops,
            actual_hops_used=actual_hops,
            nodes_traversed=len(all_entities),
            edges_traversed=edges_traversed,
            chunks_gathered=len(all_chunk_keys),
            latency_ms=elapsed,
            per_hop_stats=per_hop_stats,
            pruned_paths=pruned,
        )

        return TraversalResult(
            expanded_entities=all_entities,
            traversal_paths=all_paths,
            chunk_keys=all_chunk_keys,
            log=log,
        )

    # ── Internal neighbor accessors ───────────────────────────────────────────

    def _get_multigraph_neighbors(self, entity: str) -> List[Tuple[str, str, float]]:
        """Get neighbors from a MultiDiGraph with typed edges."""
        neighbors = []
        if entity not in self.graph:
            return neighbors

        # Outgoing edges
        for tgt in self.graph.successors(entity):
            edge_data = self.graph.get_edge_data(entity, tgt)
            if edge_data:
                # Pick the highest-weight edge among parallel edges
                best_key = max(edge_data, key=lambda k: edge_data[k].get("weight", 0))
                best = edge_data[best_key]
                rel = best.get("relation_type", "related_to")
                weight = best.get("weight", 0.5) * best.get("confidence", 1.0)
                neighbors.append((tgt, rel, weight))

        # Also check incoming edges (treat as bidirectional for retrieval)
        for src in self.graph.predecessors(entity):
            if src == entity:
                continue
            edge_data = self.graph.get_edge_data(src, entity)
            if edge_data:
                best_key = max(edge_data, key=lambda k: edge_data[k].get("weight", 0))
                best = edge_data[best_key]
                rel = best.get("relation_type", "related_to")
                weight = best.get("weight", 0.5) * best.get("confidence", 1.0)
                # Mark as reverse direction
                neighbors.append((src, f"←{rel}", weight * 0.9))

        return neighbors

    def _get_graph_neighbors(self, entity: str) -> List[Tuple[str, str, float]]:
        """Get neighbors from a plain Graph (co-occurrence, backward compat)."""
        neighbors = []
        if entity not in self.graph:
            return neighbors

        for nbr in self.graph.neighbors(entity):
            edata = self.graph.get_edge_data(entity, nbr) or {}
            weight = edata.get("weight", 1)
            # Normalize co-occurrence weight to [0, 1]
            norm_weight = min(1.0, weight / 10.0)
            neighbors.append((nbr, "co_occurs_with", norm_weight))

        return neighbors


# ─── Utility: determine suggested hops for a query classification ─────────────

def suggest_hops(query_type: str, confidence: float = 1.0) -> int:
    """
    Map query classification to suggested max_hops.

    Parameters
    ----------
    query_type : str
        One of: factual, relational, thematic, multi_hop
    confidence : float
        Classifier confidence (lower confidence → more hops as safety margin).

    Returns
    -------
    int : suggested max_hops (0–3)
    """
    base_hops = {
        "factual": 0,
        "relational": 2,
        "thematic": 1,      # Thematic uses communities, not deep hops
        "multi_hop": 3,
    }
    hops = base_hops.get(query_type, 1)

    # If confidence is low, add a safety hop
    if confidence < 0.6 and hops < 3:
        hops += 1

    return hops
