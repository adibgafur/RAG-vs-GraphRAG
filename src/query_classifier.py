"""
Query Classifier — Phase 4
============================
Classifies incoming queries to route them to the cheapest retrieval
strategy sufficient to answer well.

Two implementations:
  1. LLM classifier (primary): Few-shot prompt → (type, confidence, reasoning)
  2. Heuristic fallback: Rule-based keyword/pattern classifier

Query types:
  - factual   : Simple fact lookup → vector RAG, 0-hop
  - relational: Entity connections → local graph, 1–2 hops
  - thematic  : Synthesis / themes → global community retrieval
  - multi_hop : Complex chains     → full 2–3 hop traversal

Falls back to hybrid when confidence < 0.6.
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, Callable


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: str                  # factual / relational / thematic / multi_hop
    confidence: float                # 0.0 – 1.0
    suggested_hops: int              # 0–3
    reasoning: str                   # Why this classification was chosen
    method: str                      # "llm" or "heuristic"

    def should_fallback(self) -> bool:
        """Whether confidence is too low and should fallback to hybrid."""
        return self.confidence < 0.6

    @property
    def retrieval_mode(self) -> str:
        """Map query type to retrieval mode for the pipeline."""
        if self.should_fallback():
            return "hybrid"
        return {
            "factual": "local",
            "relational": "local",
            "thematic": "global",
            "multi_hop": "hybrid",
        }.get(self.query_type, "hybrid")


# ─── Classification Prompt ────────────────────────────────────────────────────

_CLASSIFICATION_PROMPT = """You are a query classifier for a knowledge graph retrieval system. Classify the user's question into exactly one category.

Categories:
- FACTUAL: Simple fact lookup about a specific entity, person, or event. Can be answered from a single passage.
  Examples: "Who is Jensen Huang?", "What is CUDA?", "When was OpenAI founded?"

- RELATIONAL: Questions about connections, comparisons, or relationships between two or more entities. Requires linking entities across contexts.
  Examples: "How are NVIDIA and CUDA related?", "What's the connection between Elon Musk and OpenAI?", "How do Sam Altman and Jensen Huang compare on AGI?"

- THEMATIC: Broad synthesis questions about trends, themes, or patterns across multiple sources. Requires aggregating information.
  Examples: "What are the main AI safety themes across all episodes?", "How have views on AGI evolved?", "What common perspectives emerge?"

- MULTI_HOP: Complex questions requiring chaining multiple facts or traversing through several intermediate entities to arrive at an answer.
  Examples: "Which companies that invested in AI also competed with organizations founded by guests?", "How did the technology developed by NVIDIA enable the work discussed by Sam Altman?"

Respond with ONLY a JSON object:
{{"type": "<FACTUAL|RELATIONAL|THEMATIC|MULTI_HOP>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}

QUESTION: {question}

OUTPUT:"""


# ─── Query Classifier ────────────────────────────────────────────────────────

class QueryClassifier:
    """
    Classifies queries to route them to optimal retrieval strategies.

    Parameters
    ----------
    llm_fn : callable, optional
        Function(prompt: str) -> str for LLM-based classification.
        Falls back to heuristic if None or if LLM call fails.
    """

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self.llm_fn = llm_fn

    def classify(self, question: str) -> QueryClassification:
        """
        Classify a question. Tries LLM first, falls back to heuristic.
        """
        if self.llm_fn:
            try:
                result = self._classify_llm(question)
                if result is not None:
                    return result
            except Exception:
                pass

        return self._classify_heuristic(question)

    # ── LLM Classifier ───────────────────────────────────────────────────────

    def _classify_llm(self, question: str) -> Optional[QueryClassification]:
        """Use LLM few-shot prompt for classification."""
        prompt = _CLASSIFICATION_PROMPT.format(question=question)
        response = self.llm_fn(prompt)
        return self._parse_llm_response(response)

    def _parse_llm_response(self, response: str) -> Optional[QueryClassification]:
        """Parse JSON classification from LLM response."""
        response = response.strip()

        # Handle markdown code blocks
        if "```" in response:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
            if match:
                response = match.group(1).strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try finding JSON object
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not isinstance(data, dict):
            return None

        qtype = str(data.get("type", "")).strip().lower()
        confidence = float(data.get("confidence", 0.5))
        reasoning = str(data.get("reasoning", ""))

        # Normalize type
        valid_types = {"factual", "relational", "thematic", "multi_hop"}
        if qtype not in valid_types:
            return None

        confidence = max(0.0, min(1.0, confidence))

        # Map type to suggested hops
        hop_map = {"factual": 0, "relational": 2, "thematic": 1, "multi_hop": 3}
        hops = hop_map.get(qtype, 1)

        # Lower confidence → add safety hop
        if confidence < 0.6 and hops < 3:
            hops += 1

        return QueryClassification(
            query_type=qtype,
            confidence=confidence,
            suggested_hops=hops,
            reasoning=reasoning,
            method="llm",
        )

    # ── Heuristic Classifier ──────────────────────────────────────────────────

    def _classify_heuristic(self, question: str) -> QueryClassification:
        """
        Rule-based classification using keyword patterns.
        Used as fallback when LLM is unavailable or fails.
        """
        q = question.lower().strip()

        # ── THEMATIC patterns ────────────────────────────────────────────────
        thematic_patterns = [
            r"(?:main|recurring|common|overall|general)\s+(?:theme|topic|pattern|perspective)",
            r"across\s+(?:all|the|both|these|multiple|different)\s+",
            r"(?:how\s+(?:do|did|have))\s+.*(?:collectively|together|overall)",
            r"what\s+(?:common|shared|recurring)\s+",
            r"(?:themes?|trends?|patterns?)\s+(?:across|emerge|throughout)",
            r"high[- ]level\s+(?:overview|summary|view)",
            r"(?:evolve|evolution|change)\s+(?:over\s+time|across)",
            r"synthesis|synthesize|summarize\s+(?:across|the)",
        ]
        for pattern in thematic_patterns:
            if re.search(pattern, q):
                return QueryClassification(
                    query_type="thematic",
                    confidence=0.75,
                    suggested_hops=1,
                    reasoning=f"Matched thematic pattern: {pattern}",
                    method="heuristic",
                )

        # ── MULTI_HOP patterns ───────────────────────────────────────────────
        multi_hop_patterns = [
            r"which\s+.*\s+that\s+.*\s+also\s+",
            r"how\s+did\s+.*\s+(?:enable|lead\s+to|result\s+in|cause)\s+.*\s+(?:that|which)\s+",
            r"trace\s+(?:the\s+)?(?:path|connection|chain|link)",
            r"through\s+(?:what|which)\s+(?:intermediate|steps|entities)",
            r"chain\s+of\s+(?:events?|connections?|relationships?)",
        ]
        for pattern in multi_hop_patterns:
            if re.search(pattern, q):
                return QueryClassification(
                    query_type="multi_hop",
                    confidence=0.7,
                    suggested_hops=3,
                    reasoning=f"Matched multi-hop pattern: {pattern}",
                    method="heuristic",
                )

        # ── RELATIONAL patterns ──────────────────────────────────────────────
        relational_patterns = [
            r"(?:how\s+(?:is|are|was|were))\s+.*\s+(?:related|connected|linked)\s+(?:to)?",
            r"(?:connection|relationship|link)\s+(?:between|among)",
            r"(?:compare|comparison|contrast|differ)\s+.*\s+(?:views?|opinions?|perspectives?)",
            r"(?:and|between|among)\s+.*\s+(?:connected|related|linked)",
            r"how\s+(?:do|did|does)\s+.*\s+(?:compare|differ|relate)",
            r"what\s+(?:connections?|relationships?|links?)\s+(?:exist|are\s+there)",
        ]
        for pattern in relational_patterns:
            if re.search(pattern, q):
                return QueryClassification(
                    query_type="relational",
                    confidence=0.75,
                    suggested_hops=2,
                    reasoning=f"Matched relational pattern: {pattern}",
                    method="heuristic",
                )

        # Count entities (names, orgs) mentioned → relational if multiple
        # Simple heuristic: count capitalized multi-word phrases
        cap_phrases = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", question)
        if len(cap_phrases) >= 2:
            return QueryClassification(
                query_type="relational",
                confidence=0.6,
                suggested_hops=2,
                reasoning=f"Multiple named entities detected: {cap_phrases[:3]}",
                method="heuristic",
            )

        # ── FACTUAL patterns (default for simple questions) ──────────────────
        factual_patterns = [
            r"^(?:who|what|when|where)\s+(?:is|are|was|were|did)\s+",
            r"^(?:tell|describe|explain)\s+(?:me\s+)?(?:about|what)\s+",
            r"^what\s+does\s+.*\s+(?:say|think|believe|mean)\s+(?:about)?\s+",
        ]
        for pattern in factual_patterns:
            if re.search(pattern, q):
                return QueryClassification(
                    query_type="factual",
                    confidence=0.7,
                    suggested_hops=0,
                    reasoning=f"Matched factual pattern: {pattern}",
                    method="heuristic",
                )

        # Default: factual with moderate confidence
        return QueryClassification(
            query_type="factual",
            confidence=0.5,
            suggested_hops=1,
            reasoning="No strong pattern match; defaulting to factual with 1-hop safety.",
            method="heuristic",
        )


# ─── Classification badge for UI ─────────────────────────────────────────────

def classification_badge_html(classification: QueryClassification) -> str:
    """Render a styled HTML badge for the classification result."""
    type_config = {
        "factual":    {"color": "#11998e", "icon": "🎯", "label": "Factual"},
        "relational": {"color": "#6366f1", "icon": "🔗", "label": "Relational"},
        "thematic":   {"color": "#f59221", "icon": "🌐", "label": "Thematic"},
        "multi_hop":  {"color": "#ef4444", "icon": "🔀", "label": "Multi-Hop"},
    }
    cfg = type_config.get(classification.query_type, type_config["factual"])
    conf_pct = int(classification.confidence * 100)

    method_label = "LLM" if classification.method == "llm" else "Heuristic"

    return (
        f'<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;margin:0.3rem 0">'
        f'<span style="background:{cfg["color"]}22;color:{cfg["color"]};'
        f'border-radius:8px;padding:3px 10px;font-size:0.8rem;font-weight:600">'
        f'{cfg["icon"]} {cfg["label"]}</span>'
        f'<span style="color:#888;font-size:0.75rem">'
        f'Confidence: {conf_pct}% · {classification.suggested_hops} hop{"s" if classification.suggested_hops != 1 else ""}'
        f' · via {method_label}</span>'
        f'</div>'
    )
