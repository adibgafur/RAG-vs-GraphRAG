"""
Reasoning Path — Phase 1
=========================
Builds human-readable traversal explanations showing *why* the system
retrieved what it retrieved.

Each answer is accompanied by a chain of (entity → relation → entity) hops
with supporting chunk evidence at each step.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import html


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """One hop in the reasoning chain."""
    source_entity: str
    relation: str
    target_entity: str
    confidence: float = 1.0
    supporting_chunk_ids: List[str] = field(default_factory=list)
    hop_depth: int = 0


@dataclass
class ReasoningPath:
    """Full reasoning path from query to answer."""
    query: str
    query_type: str                          # factual / relational / thematic / multi_hop
    steps: List[ReasoningStep] = field(default_factory=list)
    seed_entities: List[str] = field(default_factory=list)
    expanded_entities: List[str] = field(default_factory=list)
    total_hops: int = 0
    communities_used: List[int] = field(default_factory=list)
    all_chunk_ids: List[str] = field(default_factory=list)


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_reasoning_path(
    query: str,
    query_type: str,
    graph,                                    # NetworkX MultiDiGraph or Graph
    seed_entities: List[str],
    traversed_entities: List[str],
    traversal_paths: Optional[List[Tuple]] = None,
    chunk_ids: Optional[List[str]] = None,
    entity_to_chunks: Optional[Dict[str, set]] = None,
    communities_used: Optional[List[int]] = None,
) -> ReasoningPath:
    """
    Construct a ReasoningPath from the retrieval traversal.

    Parameters
    ----------
    query : str
        The original user query.
    query_type : str
        Classified query type.
    graph : nx.MultiDiGraph | nx.Graph
        The knowledge graph (typed edges if MultiDiGraph).
    seed_entities : list[str]
        Initial entities extracted from query + top chunks.
    traversed_entities : list[str]
        All entities reached during traversal (superset of seed).
    traversal_paths : list[tuple], optional
        Explicit (src, rel, tgt, weight, hop) tuples from MultiHopRetriever.
    chunk_ids : list[str], optional
        All chunk IDs gathered as evidence.
    entity_to_chunks : dict, optional
        Entity → set of chunk keys mapping.
    communities_used : list[int], optional
        Community IDs used in global/hybrid retrieval.
    """
    path = ReasoningPath(
        query=query,
        query_type=query_type,
        seed_entities=list(seed_entities),
        expanded_entities=list(traversed_entities),
        communities_used=communities_used or [],
        all_chunk_ids=chunk_ids or [],
    )

    # If we have explicit traversal paths from multi-hop retriever, use them
    if traversal_paths:
        for src, rel, tgt, weight, hop in traversal_paths:
            supporting = []
            if entity_to_chunks:
                src_chunks = entity_to_chunks.get(src, set())
                tgt_chunks = entity_to_chunks.get(tgt, set())
                supporting = sorted(src_chunks & tgt_chunks)[:3]

            path.steps.append(ReasoningStep(
                source_entity=src,
                relation=rel,
                target_entity=tgt,
                confidence=weight,
                supporting_chunk_ids=supporting,
                hop_depth=hop,
            ))
        path.total_hops = max((s.hop_depth for s in path.steps), default=0)
        return path

    # Otherwise, reconstruct from graph edges between seed → expanded entities
    import networkx as nx
    all_ents = set(seed_entities) | set(traversed_entities)

    if isinstance(graph, nx.MultiDiGraph):
        # Typed edges available
        for src in seed_entities:
            if src not in graph:
                continue
            for tgt in graph.successors(src):
                if tgt in all_ents and tgt != src:
                    # Pick highest-weight edge
                    edge_data = graph.get_edge_data(src, tgt)
                    if edge_data:
                        best_key = max(edge_data, key=lambda k: edge_data[k].get("weight", 0))
                        edata = edge_data[best_key]
                        supporting = []
                        if entity_to_chunks:
                            s_chunks = entity_to_chunks.get(src, set())
                            t_chunks = entity_to_chunks.get(tgt, set())
                            supporting = sorted(s_chunks & t_chunks)[:3]

                        path.steps.append(ReasoningStep(
                            source_entity=src,
                            relation=edata.get("relation_type", "related_to"),
                            target_entity=tgt,
                            confidence=edata.get("weight", 1.0),
                            supporting_chunk_ids=supporting,
                            hop_depth=1,
                        ))
    elif isinstance(graph, nx.Graph):
        # Flat co-occurrence graph (backward compat with original GraphRAG)
        for src in seed_entities:
            if src not in graph:
                continue
            for tgt in graph.neighbors(src):
                if tgt in all_ents and tgt != src:
                    edata = graph.get_edge_data(src, tgt) or {}
                    supporting = []
                    if entity_to_chunks:
                        s_chunks = entity_to_chunks.get(src, set())
                        t_chunks = entity_to_chunks.get(tgt, set())
                        supporting = sorted(s_chunks & t_chunks)[:3]

                    path.steps.append(ReasoningStep(
                        source_entity=src,
                        relation="co_occurs_with",
                        target_entity=tgt,
                        confidence=edata.get("weight", 1.0),
                        supporting_chunk_ids=supporting,
                        hop_depth=1,
                    ))

    # Deduplicate steps (same src→tgt)
    seen = set()
    unique_steps = []
    for step in path.steps:
        key = (step.source_entity, step.target_entity)
        if key not in seen:
            seen.add(key)
            unique_steps.append(step)
    path.steps = unique_steps[:20]  # Cap to avoid huge paths
    path.total_hops = max((s.hop_depth for s in path.steps), default=0)

    return path


# ─── Formatters ───────────────────────────────────────────────────────────────

def format_reasoning_path(path: ReasoningPath) -> str:
    """Format as a plain-text reasoning path."""
    if not path.steps:
        if path.communities_used:
            return (
                f"Query Type: {path.query_type}\n"
                f"Routed to community-based retrieval (communities: {path.communities_used})\n"
                f"Seed entities: {', '.join(path.seed_entities[:10])}"
            )
        return (
            f"Query Type: {path.query_type}\n"
            f"Direct vector retrieval (0-hop)\n"
            f"Seed entities: {', '.join(path.seed_entities[:10])}"
        )

    lines = [
        f"Query Type: {path.query_type}",
        f"Hops Used: {path.total_hops}",
        f"Seed Entities: {', '.join(path.seed_entities[:8])}",
        "",
        "Reasoning Chain:",
    ]

    for i, step in enumerate(path.steps, 1):
        arrow = f"  {step.source_entity} --[{step.relation}]--> {step.target_entity}"
        conf = f"  (confidence: {step.confidence:.2f}, hop: {step.hop_depth})"
        chunks = ""
        if step.supporting_chunk_ids:
            chunks = f"  Evidence: [{', '.join(step.supporting_chunk_ids[:3])}]"
        lines.append(f"{i}. {arrow}")
        lines.append(f"   {conf}")
        if chunks:
            lines.append(f"   {chunks}")

    lines.append(f"\nTotal chunks gathered: {len(path.all_chunk_ids)}")
    return "\n".join(lines)


def format_reasoning_path_html(path: ReasoningPath) -> str:
    """Format as styled HTML for Streamlit rendering."""
    esc = html.escape

    if not path.steps:
        if path.communities_used:
            comms = ", ".join(str(c) for c in path.communities_used)
            seeds = ", ".join(esc(e) for e in path.seed_entities[:10])
            return (
                f'<div style="background:#f0f4ff;border-left:4px solid #6366f1;'
                f'padding:1rem;border-radius:0 8px 8px 0;margin:0.5rem 0">'
                f'<div style="font-weight:600;color:#6366f1;margin-bottom:0.5rem">'
                f'🌐 Community Retrieval</div>'
                f'<div style="font-size:0.85rem;color:#555">'
                f'Communities used: {comms}<br>'
                f'Seed entities: {seeds}</div></div>'
            )
        seeds = ", ".join(esc(e) for e in path.seed_entities[:10])
        return (
            f'<div style="background:#f0fdf4;border-left:4px solid #11998e;'
            f'padding:1rem;border-radius:0 8px 8px 0;margin:0.5rem 0">'
            f'<div style="font-weight:600;color:#11998e;margin-bottom:0.5rem">'
            f'⚡ Direct Vector Retrieval (0-hop)</div>'
            f'<div style="font-size:0.85rem;color:#555">'
            f'Seed entities: {seeds}</div></div>'
        )

    # Build chain HTML
    chain_parts = []
    for step in path.steps[:12]:
        src = esc(step.source_entity)
        rel = esc(step.relation)
        tgt = esc(step.target_entity)
        conf_pct = int(step.confidence * 100)

        # Color code confidence
        if step.confidence >= 0.8:
            conf_color = "#11998e"
        elif step.confidence >= 0.5:
            conf_color = "#f59221"
        else:
            conf_color = "#ef4444"

        chain_parts.append(
            f'<div style="display:flex;align-items:center;gap:0.5rem;'
            f'margin:0.3rem 0;flex-wrap:wrap">'
            f'<span style="background:#11998e;color:white;border-radius:6px;'
            f'padding:2px 8px;font-size:0.8rem;font-weight:600">{src}</span>'
            f'<span style="color:#888;font-size:0.75rem">──[</span>'
            f'<span style="background:#6366f122;color:#6366f1;border-radius:4px;'
            f'padding:1px 6px;font-size:0.72rem;font-weight:500">{rel}</span>'
            f'<span style="color:#888;font-size:0.75rem">]──▶</span>'
            f'<span style="background:#11998e;color:white;border-radius:6px;'
            f'padding:2px 8px;font-size:0.8rem;font-weight:600">{tgt}</span>'
            f'<span style="color:{conf_color};font-size:0.7rem;'
            f'font-weight:600">{conf_pct}%</span>'
            f'</div>'
        )

    chain_html = "\n".join(chain_parts)

    seeds = " ".join(
        f'<span style="background:#11998e22;color:#11998e;border-radius:4px;'
        f'padding:1px 6px;font-size:0.72rem;margin:1px">{esc(e)}</span>'
        for e in path.seed_entities[:8]
    )

    type_colors = {
        "factual": "#11998e",
        "relational": "#6366f1",
        "thematic": "#f59221",
        "multi_hop": "#ef4444",
    }
    tc = type_colors.get(path.query_type, "#888")

    return (
        f'<div style="background:#fafbfc;border:1px solid #e2e8f0;'
        f'border-radius:10px;padding:1rem;margin:0.5rem 0">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem">'
        f'<span style="font-weight:700;font-size:0.9rem">🧠 Reasoning Path</span>'
        f'<span style="background:{tc}22;color:{tc};border-radius:6px;'
        f'padding:2px 8px;font-size:0.75rem;font-weight:600">'
        f'{esc(path.query_type)}</span>'
        f'<span style="color:#888;font-size:0.75rem">'
        f'{path.total_hops} hop{"s" if path.total_hops != 1 else ""} · '
        f'{len(path.all_chunk_ids)} chunks</span>'
        f'</div>'
        f'<div style="margin-bottom:0.5rem">{chain_html}</div>'
        f'<div style="border-top:1px solid #e2e8f0;padding-top:0.5rem;'
        f'margin-top:0.5rem">'
        f'<span style="font-size:0.75rem;color:#888">Seed entities: </span>'
        f'{seeds}</div>'
        f'</div>'
    )
