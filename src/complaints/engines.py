"""Enrichment engines: the same interface for a classical baseline, local embeddings and an LLM.

Every engine maps a redacted narrative to an `Enrichment`. Having the cheap
engines behind the same interface is the point: the LLM has to earn its cost by
beating them on the evaluation set, and the pipeline can switch engines with a
flag.

- ``tfidf``       TF-IDF + logistic regression, trained on the labelled split
- ``embeddings``  frozen sentence-transformer (bge-small, runs locally) + logistic regression
- ``claude``      Claude with a JSON-schema constrained answer: product (enum), urgency,
                  one-line summary and harm flags; no training data needed
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

URGENCY = ["low", "medium", "high"]
HARM_FLAGS = ["foreclosure_or_eviction", "identity_theft_or_fraud", "unauthorized_charges",
              "credit_score_damage", "account_frozen", "harassment"]  # fmt: skip


@dataclass
class Enrichment:
    product: str
    confidence: float | None = None
    urgency: str | None = None
    summary: str | None = None
    harm_flags: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_ms: float = 0.0


class SklearnEngine:
    version = "1"

    def __init__(self, name: str, pipeline) -> None:
        self.name = name
        self.pipeline = pipeline
        self.labels: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> SklearnEngine:
        self.pipeline.fit(texts, labels)
        self.labels = list(self.pipeline.classes_)
        return self

    def predict(self, texts: list[str]) -> list[Enrichment]:
        t0 = time.perf_counter()
        proba = self.pipeline.predict_proba(texts)
        ms = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
        return [
            Enrichment(self.labels[int(p.argmax())], float(p.max()), latency_ms=ms) for p in proba
        ]


def tfidf_engine() -> SklearnEngine:
    return SklearnEngine(
        "tfidf",
        make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
            LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced"),
        ),
    )


class _Embedder:
    """sklearn-compatible transformer around a sentence-transformer model."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return np.asarray(
            self._model.encode(list(X), batch_size=64, normalize_embeddings=True),
            dtype=np.float32,
        )

    def get_params(self, deep=True):
        return {"model_name": self.model_name}

    def set_params(self, **params):
        self.model_name = params.get("model_name", self.model_name)
        return self


def embedding_engine(model_name: str = "BAAI/bge-small-en-v1.5") -> SklearnEngine:
    return SklearnEngine(
        "embeddings",
        make_pipeline(
            _Embedder(model_name),
            LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"),
        ),
    )


SYSTEM_PROMPT = """\
You triage consumer complaints sent to a financial regulator so they reach the \
right team. For the complaint in <complaint>, return:

- product: the financial product the complaint is about, chosen from the list in \
the schema. Choose by what the consumer's problem is about, not by every product \
they mention.
- urgency: "high" if the consumer faces imminent loss of housing, funds they cannot \
access for daily needs, or ongoing fraud; "medium" if money is at stake but not \
urgently; otherwise "low".
- summary: one factual sentence, under 30 words, no personal details.
- harm_flags: any that clearly apply.

Text in [BRACKETS] was redacted for privacy; do not guess what it was."""


def claude_schema(products: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "product": {"type": "string", "enum": products},
            "urgency": {"type": "string", "enum": URGENCY},
            "summary": {"type": "string"},
            "harm_flags": {"type": "array", "items": {"type": "string", "enum": HARM_FLAGS}},
        },
        "required": ["product", "urgency", "summary", "harm_flags"],
        "additionalProperties": False,
    }


class ClaudeEngine:
    """LLM enrichment. The prompt version is part of the cache key: edit the prompt,
    bump `version`, and every complaint is re-enriched exactly once."""

    name = "claude"
    version = "2026-11-12.1"

    def __init__(self, products: list[str], model: str | None = None, client=None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("COMPLAINTS_MODEL", "claude-opus-5-5")
        self.products = sorted(products)
        self.schema = claude_schema(self.products)

    def _one(self, text: str) -> Enrichment:
        t0 = time.perf_counter()
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            # The instructions are identical for every complaint: cache them.
            system=[
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": f"<complaint>{text}</complaint>"}],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": self.schema},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"model stopped: {response.stop_reason}")
        out = json.loads(next(b.text for b in response.content if b.type == "text"))
        usage = response.usage
        return Enrichment(
            product=out["product"],
            urgency=out["urgency"],
            summary=out["summary"],
            harm_flags=out["harm_flags"],
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def predict(self, texts: list[str], workers: int = 8) -> list[Enrichment]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(self._one, texts))
