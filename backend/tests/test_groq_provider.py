"""Groq provider request shaping, with a stubbed client (no network or API key needed)."""

from types import SimpleNamespace

from app.core.config import get_settings
from app.services import llm


class _StubCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            chunks = ["HR 0.80 ", "(0.73-0.87) [trial.pdf p.22]"]
            return iter(
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))]) for c in chunks
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer [trial.pdf p.22]"))])


def _service(model: str) -> llm.GroqLLMService:
    svc = llm.GroqLLMService.__new__(llm.GroqLLMService)
    svc.client = SimpleNamespace(chat=SimpleNamespace(completions=_StubCompletions()))
    svc.model = model
    svc._guard_budget = lambda **_: None
    return svc


CONTEXTS = [{"citation": "trial.pdf p.22", "text": "HR 0.80; 95% CI 0.73, 0.87"}]


def test_reasoning_model_gets_low_effort_and_token_headroom():
    settings = get_settings()
    svc = _service("openai/gpt-oss-120b")
    assert svc.answer(question="q", rewritten_query="q", contexts=CONTEXTS) == "answer [trial.pdf p.22]"
    call = svc.client.chat.completions.calls[-1]
    assert call["extra_body"] == {"reasoning_effort": "low"}
    assert call["max_completion_tokens"] == settings.max_generation_tokens + settings.groq_reasoning_token_headroom

    streamed = "".join(svc.stream_answer(question="q", rewritten_query="q", contexts=CONTEXTS))
    assert streamed == "HR 0.80 (0.73-0.87) [trial.pdf p.22]"
    assert svc.client.chat.completions.calls[-1]["extra_body"] == {"reasoning_effort": "low"}


def test_non_reasoning_model_request_is_unchanged():
    settings = get_settings()
    svc = _service("some-chat-model")
    svc.answer(question="q", rewritten_query="q", contexts=CONTEXTS)
    call = svc.client.chat.completions.calls[-1]
    assert "extra_body" not in call
    assert call["max_completion_tokens"] == settings.max_generation_tokens


def test_default_model_is_not_a_retired_llama():
    assert get_settings().groq_model == "openai/gpt-oss-120b"
