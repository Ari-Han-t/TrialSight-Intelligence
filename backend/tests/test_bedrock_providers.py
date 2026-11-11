"""Bedrock provider tests with stubbed clients (no AWS access required)."""

import io
import json

import numpy as np

from app.services import embeddings, llm


class _StubBedrockRuntime:
    def __init__(self):
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        body = json.loads(kwargs["body"])
        # Deterministic fake embedding derived from text length.
        vec = [float((len(body["inputText"]) + i) % 7) for i in range(body["dimensions"])]
        return {"body": io.BytesIO(json.dumps({"embedding": vec}).encode())}

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": "Primary endpoint met "}, {"text": "[trial.pdf p.1]"}]}}}

    def converse_stream(self, **kwargs):
        self.calls.append(kwargs)
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "SBP fell "}}},
            {"contentBlockDelta": {"delta": {"text": "11.2 mmHg."}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        return {"stream": iter(events)}


def _titan(dim=16):
    emb = embeddings.BedrockTitanEmbedder.__new__(embeddings.BedrockTitanEmbedder)
    emb.client = _StubBedrockRuntime()
    emb.model_id = "amazon.titan-embed-text-v2:0"
    emb.dim = dim
    emb.name = f"bedrock:{emb.model_id}:{dim}"
    return emb


def test_titan_embedder_normalises_and_round_trips():
    emb = _titan()
    vectors = emb.embed_documents(["alpha", "beta gamma"])
    assert len(vectors) == 2
    for v in vectors:
        assert v.dtype == np.float32 and v.shape == (16,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
        assert np.array_equal(embeddings.from_bytes(embeddings.to_bytes(v)), v)
    request = json.loads(emb.client.calls[0]["body"])
    assert request["dimensions"] == 16 and request["normalize"] is True


def test_hash_embedder_is_deterministic():
    h = embeddings.HashEmbedder(64)
    assert np.array_equal(h.embed_query("XR-17 reduced SBP"), h.embed_query("XR-17 reduced SBP"))
    assert embeddings.from_bytes(None) is None


def _bedrock_llm():
    svc = llm.BedrockLLMService.__new__(llm.BedrockLLMService)
    svc.client = _StubBedrockRuntime()
    svc.model_id = "test-model"
    svc._guard_budget = lambda **_: None  # budgets are covered by the shared base class
    return svc


def test_bedrock_answer_and_stream():
    svc = _bedrock_llm()
    contexts = [{"citation": "trial.pdf p.1", "text": "SBP fell 11.2 mmHg"}]
    answer = svc.answer(question="effect?", rewritten_query="effect", contexts=contexts)
    assert answer == "Primary endpoint met [trial.pdf p.1]"
    request = svc.client.calls[-1]
    assert request["modelId"] == "test-model"
    assert "clinical trial evidence analyst" in request["system"][0]["text"]
    assert "[trial.pdf p.1]" in request["messages"][0]["content"][0]["text"]

    streamed = "".join(svc.stream_answer(question="effect?", rewritten_query="effect", contexts=contexts))
    assert streamed == "SBP fell 11.2 mmHg."


def test_provider_errors_map_to_http_status():
    svc = _bedrock_llm()

    def boom(**_):
        raise RuntimeError("ThrottlingException: Too many requests")

    svc._complete = boom
    try:
        svc.rewrite_query("q")
    except Exception as exc:  # FastAPI HTTPException
        assert getattr(exc, "status_code", None) == 429
    else:
        raise AssertionError("expected HTTPException")


def test_unavailable_provider_degrades_to_extractive_answer():
    svc = _bedrock_llm()

    def denied(**_):
        raise RuntimeError("AccessDeniedException: You don't have access to the model")

    svc._complete = denied
    svc._stream = denied
    contexts = [{"citation": "trial.pdf p.1", "text": "SBP fell 11.2 mmHg"}]
    assert svc.rewrite_query("what happened?") == "what happened?"
    answer = svc.answer(question="q", rewritten_query="q", contexts=contexts)
    assert answer.startswith("The bedrock model is unavailable") and "[trial.pdf p.1]" in answer
    streamed = "".join(svc.stream_answer(question="q", rewritten_query="q", contexts=contexts))
    assert "[trial.pdf p.1]" in streamed


def test_embedding_failure_falls_back_to_hash(monkeypatch):
    broken = _titan()

    def fail(_text):
        raise RuntimeError("AccessDeniedException")

    broken.embed_query = fail
    monkeypatch.setattr(embeddings, "get_embedder", lambda: broken)
    vectors, name = embeddings.embed_documents_safe(["a", "b"])
    assert name == embeddings.get_fallback_embedder().name and len(vectors) == 2
    assert set(embeddings.query_vectors("a")) == {embeddings.get_fallback_embedder().name}
