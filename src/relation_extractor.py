"""
Relation Extractor — Phase 2
==============================
Replaces bare co-occurrence edges with typed, weighted,
provenance-tagged relationships.

Two backends:
  1. LLM-based (primary): Few-shot prompt extracts structured relations.
  2. spaCy fallback: Dependency-parse heuristic when no LLM is available.

Edge schema:
  (source, target, relation_type, weight, confidence,
   source_doc_id, source_chunk_id, timestamp)
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple


# ─── Data Structures ──────────────────────────────────────────────────────────

# Controlled vocabulary for relation types
RELATION_TYPES = {
    "develops",         # A develops B (product, technology)
    "founded",          # A founded B (organization)
    "co_founded",       # A co-founded B
    "works_at",         # A works at B
    "leads",            # A leads/runs B
    "invests_in",       # A invests in B
    "acquired",         # A acquired B
    "competes_with",    # A competes with B
    "partners_with",    # A partners with B
    "uses",             # A uses B (technology, method)
    "proposes",         # A proposes B (idea, model)
    "evaluates",        # A evaluates B
    "criticizes",       # A criticizes B
    "supports",         # A supports B
    "improves",         # A improves B
    "discusses",        # A discusses B
    "mentions",         # A mentions B
    "related_to",       # Generic fallback relation
    "compares_with",    # A compares with B
    "part_of",          # A is part of B
    "enables",          # A enables B
    "built_on",         # A is built on B
    "trained_on",       # A is trained on B (model→dataset)
    "succeeds",         # A succeeds B (newer version)
    "co_occurs_with",   # Fallback: co-occurrence only
}


@dataclass
class Relation:
    """A single extracted relation between two entities."""
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    confidence: float = 1.0
    source_doc_id: str = ""
    source_chunk_id: str = ""
    timestamp: str = ""
    raw_text: str = ""             # snippet from source text

    def to_edge_attrs(self) -> dict:
        """Convert to NetworkX edge attribute dict."""
        return {
            "relation_type": self.relation_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "source_doc_id": self.source_doc_id,
            "source_chunk_id": self.source_chunk_id,
            "timestamp": self.timestamp,
            "raw_text": self.raw_text[:200],
        }


# ─── LLM Extraction Prompt ───────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are an expert knowledge graph builder. Extract entity relationships from the text below.

For each relationship, output a JSON object with:
- "source": the subject entity (lowercase, canonical form)
- "target": the object entity (lowercase, canonical form)
- "relation": one of: develops, founded, co_founded, works_at, leads, invests_in, acquired, competes_with, partners_with, uses, proposes, evaluates, criticizes, supports, improves, discusses, mentions, related_to, compares_with, part_of, enables, built_on, trained_on, succeeds
- "confidence": 0.0 to 1.0 (how confident you are this relation exists)

Output ONLY a JSON array of objects. No explanation. If no relations found, output [].

Examples:
Text: "Jensen Huang, who founded NVIDIA, discussed how CUDA enables GPU computing."
Output: [{{"source":"jensen huang","target":"nvidia","relation":"founded","confidence":0.95}},{{"source":"nvidia","target":"cuda","relation":"develops","confidence":0.9}},{{"source":"cuda","target":"gpu computing","relation":"enables","confidence":0.85}}]

Text: "Sam Altman leads OpenAI, which developed GPT-4 and ChatGPT."
Output: [{{"source":"sam altman","target":"openai","relation":"leads","confidence":0.95}},{{"source":"openai","target":"gpt-4","relation":"develops","confidence":0.95}},{{"source":"openai","target":"chatgpt","relation":"develops","confidence":0.95}}]

Now extract relations from:
TEXT: {text}

OUTPUT:"""


# ─── Relation Extractor ──────────────────────────────────────────────────────

class RelationExtractor:
    """
    Extracts typed, weighted relationships from text chunks.

    Parameters
    ----------
    llm_fn : callable, optional
        Function(prompt: str) -> str for LLM-based extraction.
        If None, falls back to spaCy dependency-parse heuristic.
    spacy_model : str
        spaCy model name for entity extraction and fallback.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        spacy_model: str = "en_core_web_sm"
    ):
        self.llm_fn = llm_fn
        self._nlp = None
        self._spacy_model = spacy_model
        self._entity_labels = {
            "PERSON", "ORG", "GPE", "PRODUCT", "EVENT",
            "WORK_OF_ART", "NORP", "LAW", "FAC"
        }

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self._spacy_model)
            except Exception:
                self._nlp = "regex_fallback"
        return self._nlp

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_relations(
        self,
        text: str,
        source_doc_id: str = "",
        source_chunk_id: str = "",
        timestamp: str = "",
    ) -> List[Relation]:
        """
        Extract relations from a single text chunk.
        Uses LLM if available, otherwise falls back to spaCy.
        """
        if self.llm_fn:
            return self._extract_llm(text, source_doc_id, source_chunk_id, timestamp)
        else:
            return self._extract_spacy(text, source_doc_id, source_chunk_id, timestamp)

    def extract_entities(self, text: str) -> List[str]:
        """Return unique lowercase entity strings from text (spaCy NER or regex fallback)."""
        nlp = self._get_nlp()
        if nlp == "regex_fallback":
            matches = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b", text)
            seen, entities = set(), []
            for m in matches:
                name = m.strip().lower()
                if len(name) > 2 and not re.match(r"^\d+[\d\s,\.]*$", name):
                    if name not in seen:
                        seen.add(name)
                        entities.append(name)
            return entities

        doc = nlp(text)
        seen, entities = set(), []
        for ent in doc.ents:
            if ent.label_ in self._entity_labels:
                name = ent.text.strip().lower()
                if len(name) > 2 and not re.match(r"^\d+[\d\s,\.]*$", name):
                    if name not in seen:
                        seen.add(name)
                        entities.append(name)
        return entities

    def extract_batch(
        self,
        chunks: list,
        source_name: str = "",
    ) -> List[Relation]:
        """Extract relations from a list of TextChunk objects."""
        all_relations = []
        for chunk in chunks:
            chunk_id = f"{getattr(chunk, 'source', source_name)}_{getattr(chunk, 'chunk_id', 0)}"
            rels = self.extract_relations(
                text=chunk.text,
                source_doc_id=getattr(chunk, "source", source_name),
                source_chunk_id=chunk_id,
                timestamp=getattr(chunk, "start_time", ""),
            )
            all_relations.extend(rels)
        return all_relations

    # ── LLM-Based Extraction ──────────────────────────────────────────────────

    def _extract_llm(
        self, text: str,
        source_doc_id: str, source_chunk_id: str, timestamp: str
    ) -> List[Relation]:
        """Use LLM few-shot prompt to extract structured relations."""
        text_trunc = text[:1500] if len(text) > 1500 else text
        prompt = _EXTRACTION_PROMPT.format(text=text_trunc)

        try:
            response = self.llm_fn(prompt)
            relations = self._parse_llm_response(
                response, text, source_doc_id, source_chunk_id, timestamp
            )
            if not relations:
                relations = self._extract_spacy(
                    text, source_doc_id, source_chunk_id, timestamp
                )
            return relations
        except Exception:
            return self._extract_spacy(
                text, source_doc_id, source_chunk_id, timestamp
            )

    def _parse_llm_response(
        self, response: str, text: str,
        source_doc_id: str, source_chunk_id: str, timestamp: str
    ) -> List[Relation]:
        """Parse JSON array from LLM response."""
        response = response.strip()

        if "```" in response:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
            if match:
                response = match.group(1).strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        relations = []
        for item in data:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", "")).strip().lower()
            tgt = str(item.get("target", "")).strip().lower()
            rel = str(item.get("relation", "related_to")).strip().lower()
            conf = float(item.get("confidence", 0.8))

            if not src or not tgt or src == tgt:
                continue
            if len(src) < 2 or len(tgt) < 2:
                continue

            rel = rel.replace(" ", "_").replace("-", "_")
            if rel not in RELATION_TYPES:
                rel = "related_to"

            conf = max(0.0, min(1.0, conf))

            relations.append(Relation(
                source=src,
                target=tgt,
                relation_type=rel,
                weight=conf,
                confidence=conf,
                source_doc_id=source_doc_id,
                source_chunk_id=source_chunk_id,
                timestamp=timestamp,
                raw_text=text[:200],
            ))

        return relations

    # ── spaCy Fallback ────────────────────────────────────────────────────────

    def _extract_spacy(
        self, text: str,
        source_doc_id: str, source_chunk_id: str, timestamp: str
    ) -> List[Relation]:
        """
        Fallback: extract entities via spaCy NER (or regex) and create co-occurrence
        relations with simple dependency-parse or heuristic relation typing.
        """
        nlp = self._get_nlp()
        if nlp == "regex_fallback":
            ents = self.extract_entities(text)
            if len(ents) < 2:
                return []
            relations = []
            for i, e1 in enumerate(ents):
                for e2 in ents[i + 1:]:
                    if e1 != e2:
                        relations.append(Relation(
                            source=e1,
                            target=e2,
                            relation_type="co_occurs_with",
                            weight=0.5,
                            confidence=0.5,
                            source_doc_id=source_doc_id,
                            source_chunk_id=source_chunk_id,
                            timestamp=timestamp,
                            raw_text=text[:200],
                        ))
            return relations

        doc = nlp(text)

        # Collect entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in self._entity_labels:
                name = ent.text.strip().lower()
                if len(name) > 2 and not re.match(r"^\d+[\d\s,\.]*$", name):
                    entities.append({
                        "text": name,
                        "label": ent.label_,
                        "start": ent.start,
                        "end": ent.end,
                    })

        if len(entities) < 2:
            return []

        relations = []

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if e1["text"] == e2["text"]:
                    continue

                start_tok = min(e1["end"], e2["end"])
                end_tok = max(e1["start"], e2["start"])

                rel_type = "co_occurs_with"
                confidence = 0.5

                for token in doc[start_tok:end_tok]:
                    if token.pos_ == "VERB":
                        verb = token.lemma_.lower()
                        rel_type = self._verb_to_relation(verb)
                        if rel_type != "co_occurs_with":
                            confidence = 0.65
                        break

                if e1["label"] == "PERSON" and e2["label"] == "ORG":
                    if rel_type == "co_occurs_with":
                        rel_type = "related_to"
                        confidence = 0.55

                relations.append(Relation(
                    source=e1["text"],
                    target=e2["text"],
                    relation_type=rel_type,
                    weight=confidence,
                    confidence=confidence,
                    source_doc_id=source_doc_id,
                    source_chunk_id=source_chunk_id,
                    timestamp=timestamp,
                    raw_text=text[:200],
                ))

        return relations

    def _verb_to_relation(self, verb: str) -> str:
        """Map common verbs to relation types."""
        verb_map = {
            "develop": "develops",
            "build": "develops",
            "create": "develops",
            "design": "develops",
            "found": "founded",
            "start": "founded",
            "lead": "leads",
            "run": "leads",
            "head": "leads",
            "manage": "leads",
            "invest": "invests_in",
            "fund": "invests_in",
            "acquire": "acquired",
            "buy": "acquired",
            "purchase": "acquired",
            "compete": "competes_with",
            "rival": "competes_with",
            "partner": "partners_with",
            "collaborate": "partners_with",
            "use": "uses",
            "employ": "uses",
            "utilize": "uses",
            "leverage": "uses",
            "propose": "proposes",
            "suggest": "proposes",
            "introduce": "proposes",
            "evaluate": "evaluates",
            "test": "evaluates",
            "benchmark": "evaluates",
            "assess": "evaluates",
            "criticize": "criticizes",
            "oppose": "criticizes",
            "challenge": "criticizes",
            "support": "supports",
            "endorse": "supports",
            "advocate": "supports",
            "improve": "improves",
            "enhance": "improves",
            "upgrade": "improves",
            "optimize": "improves",
            "discuss": "discusses",
            "describe": "discusses",
            "explain": "discusses",
            "mention": "mentions",
            "refer": "mentions",
            "cite": "mentions",
            "compare": "compares_with",
            "enable": "enables",
            "allow": "enables",
            "power": "enables",
            "succeed": "succeeds",
            "replace": "succeeds",
            "train": "trained_on",
        }
        return verb_map.get(verb, "co_occurs_with")


# ─── Utility: Merge duplicate relations ───────────────────────────────────────

def merge_relations(relations: List[Relation]) -> List[Relation]:
    """
    Merge duplicate (source, target, relation_type) triples by
    averaging weights and collecting all source chunks.
    """
    merged: Dict[Tuple[str, str, str], dict] = {}

    for rel in relations:
        key = (rel.source, rel.target, rel.relation_type)
        if key not in merged:
            merged[key] = {
                "relation": rel,
                "weights": [rel.weight],
                "confidences": [rel.confidence],
                "chunk_ids": {rel.source_chunk_id},
                "count": 1,
            }
        else:
            merged[key]["weights"].append(rel.weight)
            merged[key]["confidences"].append(rel.confidence)
            merged[key]["chunk_ids"].add(rel.source_chunk_id)
            merged[key]["count"] += 1

    result = []
    for key, data in merged.items():
        rel = data["relation"]
        # Weight increases with frequency (more mentions = stronger relation)
        avg_conf = sum(data["confidences"]) / len(data["confidences"])
        freq_boost = min(1.2, 1.0 + 0.05 * (data["count"] - 1))
        rel.weight = round(min(1.0, avg_conf * freq_boost), 3)
        rel.confidence = round(avg_conf, 3)
        result.append(rel)

    return sorted(result, key=lambda r: r.weight, reverse=True)
