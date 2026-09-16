"""
Unit Tests for Adaptive Multi-Hop GraphRAG Framework
=====================================================
Tests:
  - QueryClassifier
  - RelationExtractor
  - MultiHopRetriever
  - ReasoningPath
  - AdaptiveGraphRAG
"""

import sys
import unittest
import shutil
import tempfile
from pathlib import Path
import networkx as nx

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.query_classifier import QueryClassifier, QueryClassification, classification_badge_html
from src.relation_extractor import RelationExtractor, Relation, merge_relations
from src.multi_hop_retriever import MultiHopRetriever, TraversalLog, suggest_hops
from src.reasoning_path import (
    ReasoningPath,
    ReasoningStep,
    build_reasoning_path,
    format_reasoning_path,
    format_reasoning_path_html,
)
from src.adaptive_pipeline import AdaptiveGraphRAG, QueryLog, TextChunk


class TestQueryClassifier(unittest.TestCase):
    """Test heuristic and LLM query classification."""

    def setUp(self):
        self.classifier = QueryClassifier()

    def test_factual_classification(self):
        result = self.classifier.classify("Who is Jensen Huang?")
        self.assertIsInstance(result, QueryClassification)
        self.assertEqual(result.query_type, "factual")
        self.assertEqual(result.suggested_hops, 0)
        self.assertFalse(result.should_fallback())

    def test_relational_classification(self):
        result = self.classifier.classify("How are Jensen Huang and OpenAI connected?")
        self.assertIsInstance(result, QueryClassification)
        self.assertIn(result.query_type, ("relational", "multi_hop"))
        self.assertGreaterEqual(result.suggested_hops, 2)

    def test_thematic_classification(self):
        result = self.classifier.classify("What are the main AI safety themes across all episodes?")
        self.assertIsInstance(result, QueryClassification)
        self.assertEqual(result.query_type, "thematic")

    def test_multi_hop_classification(self):
        result = self.classifier.classify(
            "Which companies that invested in AI also competed with organizations founded by guests?"
        )
        self.assertIsInstance(result, QueryClassification)
        self.assertEqual(result.query_type, "multi_hop")
        self.assertEqual(result.suggested_hops, 3)

    def test_mock_llm_classification(self):
        def mock_llm(prompt):
            return '{"type": "relational", "confidence": 0.9, "reasoning": "Compares two entities"}'

        classifier_llm = QueryClassifier(llm_fn=mock_llm)
        result = classifier_llm.classify("How does CUDA compare to Tensor Cores?")
        self.assertEqual(result.query_type, "relational")
        self.assertEqual(result.method, "llm")
        self.assertEqual(result.confidence, 0.9)


class TestRelationExtractor(unittest.TestCase):
    """Test entity and relation extraction."""

    def setUp(self):
        self.extractor = RelationExtractor()

    def test_entity_extraction(self):
        text = "Jensen Huang, CEO of NVIDIA, announced new GPU architectures in Santa Clara."
        entities = self.extractor.extract_entities(text)
        self.assertTrue(any("jensen huang" in e for e in entities))
        self.assertTrue(any("nvidia" in e for e in entities))

    def test_spacy_fallback_extraction(self):
        text = "Jensen Huang founded NVIDIA. NVIDIA develops CUDA."
        relations = self.extractor.extract_relations(text, source_doc_id="doc1", source_chunk_id="c1")
        self.assertTrue(len(relations) >= 1)
        for rel in relations:
            self.assertIsInstance(rel, Relation)
            self.assertIsNotNone(rel.source)
            self.assertIsNotNone(rel.target)

    def test_mock_llm_extraction(self):
        def mock_llm(prompt):
            return '[{"source": "jensen huang", "target": "nvidia", "relation": "founded", "confidence": 0.95}]'

        extractor = RelationExtractor(llm_fn=mock_llm)
        relations = extractor.extract_relations("Jensen Huang founded NVIDIA.", source_chunk_id="c1")
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].source, "jensen huang")
        self.assertEqual(relations[0].target, "nvidia")
        self.assertEqual(relations[0].relation_type, "founded")

    def test_merge_relations(self):
        r1 = Relation("jensen huang", "nvidia", "founded", weight=0.8, confidence=0.8, source_chunk_id="c1")
        r2 = Relation("jensen huang", "nvidia", "founded", weight=0.9, confidence=0.9, source_chunk_id="c2")
        merged = merge_relations([r1, r2])
        self.assertEqual(len(merged), 1)
        self.assertGreater(merged[0].weight, 0.8)


class TestMultiHopRetriever(unittest.TestCase):
    """Test 0–3 hop graph traversal and score decay."""

    def setUp(self):
        self.graph = nx.MultiDiGraph()
        # Seed graph: A -> B -> C -> D
        self.graph.add_edge("jensen huang", "nvidia", relation_type="leads", weight=0.9)
        self.graph.add_edge("nvidia", "cuda", relation_type="develops", weight=0.8)
        self.graph.add_edge("cuda", "gpu computing", relation_type="enables", weight=0.7)

        self.entity_to_chunks = {
            "jensen huang": {"chunk_1"},
            "nvidia": {"chunk_1", "chunk_2"},
            "cuda": {"chunk_2", "chunk_3"},
            "gpu computing": {"chunk_3"},
        }

        self.retriever = MultiHopRetriever(
            graph=self.graph,
            entity_to_chunks=self.entity_to_chunks,
            decay_factor=0.7,
            relevance_threshold=0.2,
        )

    def test_0_hop_traversal(self):
        res = self.retriever.retrieve(seed_entities=["jensen huang"], max_hops=0)
        self.assertEqual(res.log.actual_hops_used, 0)
        self.assertIn("jensen huang", res.expanded_entities)
        self.assertEqual(res.chunk_keys, {"chunk_1"})

    def test_1_hop_traversal(self):
        res = self.retriever.retrieve(seed_entities=["jensen huang"], max_hops=1)
        self.assertGreaterEqual(res.log.actual_hops_used, 1)
        self.assertIn("nvidia", res.expanded_entities)
        self.assertIn("chunk_2", res.chunk_keys)

    def test_2_hop_traversal(self):
        res = self.retriever.retrieve(seed_entities=["jensen huang"], max_hops=2)
        self.assertIn("cuda", res.expanded_entities)
        self.assertIn("chunk_3", res.chunk_keys)


class TestReasoningPath(unittest.TestCase):
    """Test reasoning path builder and formatters."""

    def test_reasoning_path_formatting(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("sam altman", "openai", relation_type="leads", weight=0.95)

        entity_to_chunks = {
            "sam altman": {"c1"},
            "openai": {"c1", "c2"},
        }

        path = build_reasoning_path(
            query="What is Sam Altman's role at OpenAI?",
            query_type="relational",
            graph=graph,
            seed_entities=["sam altman"],
            traversed_entities=["sam altman", "openai"],
            entity_to_chunks=entity_to_chunks,
        )

        self.assertEqual(path.query_type, "relational")
        self.assertGreaterEqual(len(path.steps), 1)

        text_out = format_reasoning_path(path)
        self.assertIn("Reasoning Chain:", text_out)
        self.assertIn("sam altman", text_out)

        html_out = format_reasoning_path_html(path)
        self.assertIn("Reasoning Path", html_out)
        self.assertIn("sam altman", html_out)


class TestAdaptiveGraphRAG(unittest.TestCase):
    """Test AdaptiveGraphRAG end-to-end functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rag = AdaptiveGraphRAG(persist_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ingest_and_query(self):
        chunks = [
            TextChunk(
                chunk_id=0,
                text="Jensen Huang founded NVIDIA in 1993. NVIDIA developed CUDA for accelerated parallel computing.",
                start_time="00:00:00",
                end_time="00:00:30",
                char_start=0,
                char_end=100,
                source="episode1",
            ),
            TextChunk(
                chunk_id=1,
                text="Sam Altman leads OpenAI. OpenAI relies on NVIDIA GPUs for training massive GPT models.",
                start_time="00:00:30",
                end_time="00:01:00",
                char_start=101,
                char_end=200,
                source="episode1",
            ),
        ]

        stats = self.rag.ingest_chunks(chunks, source_name="episode1")
        self.assertEqual(stats["chunks"], 2)
        self.assertGreater(self.rag.graph.number_of_nodes(), 0)

        # Query
        res = self.rag.query("How are NVIDIA and OpenAI connected?", mode="auto")
        self.assertIn("question", res)
        self.assertIn("reasoning_path", res)
        self.assertIsInstance(res["query_log"], QueryLog)
        self.assertIsNotNone(res["reasoning_path_html"])


if __name__ == "__main__":
    unittest.main()
