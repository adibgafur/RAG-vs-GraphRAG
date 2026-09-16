"""
Adaptive Multi-Hop ResearchGraphRAG
====================================
Query-adaptive, multi-hop graph retrieval with typed relationships
and transparent reasoning paths.

Modules:
  - query_classifier    : Classify queries as factual/relational/thematic/multi_hop
  - multi_hop_retriever : Adaptive 0–3 hop graph traversal with relevance decay
  - relation_extractor  : Extract typed, weighted, provenance-tagged relationships
  - reasoning_path      : Build human-readable traversal explanations
  - adaptive_pipeline   : Main pipeline combining all features
"""

from src.query_classifier import QueryClassifier, QueryClassification
from src.multi_hop_retriever import MultiHopRetriever, TraversalLog
from src.relation_extractor import RelationExtractor, Relation
from src.reasoning_path import ReasoningPath, ReasoningStep, build_reasoning_path, format_reasoning_path
from src.adaptive_pipeline import AdaptiveGraphRAG, QueryLog
