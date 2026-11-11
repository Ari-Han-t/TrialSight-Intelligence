"""LLM orchestration with provider-agnostic budget guards.

Providers:
- ``bedrock`` : Anthropic Claude (or any Converse-compatible model) on Amazon Bedrock. Uses the
               standard AWS credential chain, so on EC2/ECS it authenticates through the attached
               IAM role and no API key ever lives in the app's environment.
- ``groq``    : Groq-hosted Llama models via API key.
- fallback    : extractive answer from the top passages when no provider is configured.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Generator, Iterator

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.rag.chunking import estimate_tokens
from app.rate_limit.store import rate_limit_store


logger = logging.getLogger(__name__)
settings = get_settings()
provider_semaphore = threading.BoundedSemaphore(value=settings.groq_max_concurrent_requests)

REWRITE_SYSTEM = (
    "Rewrite the user's question into a concise retrieval query for searching clinical trial documents "
    "(protocols, CSRs, publications). Keep drug names, populations, endpoints, arms, doses, timepoints and "
    "statistical terms. Return only the rewritten query."
)
ANSWER_SYSTEM = (
    "You are a clinical trial evidence analyst. Answer ONLY from the supplied passages. "
    "Be concise and precise with numbers (effect sizes, CIs, p-values, n). Cite every key statement "
    "with its passage label, e.g. [trial.pdf p.4]. If the passages do not contain the answer, say so "
    "plainly instead of guessing, and note any uncertainty or limitations."
)
SCORE_SYSTEM = (
    "Score the answer quality from 0 to 1 as JSON with a single key `score`, based only on factual "
    "support from the provided context. Return only the JSON object."
)


class BaseLLMService:
    provider = "none"

    def rewrite_query(self, question: str) -> str:
        return question

    def answer(self, *, question: str, rewritten_query: str, contexts: list[dict]) -> str:
        raise NotImplementedError

    def stream_answer(self, *, question: str, rewritten_query: str, contexts: list[dict]) -> Generator[str, None, None]:
        yield self.answer(question=question, rewritten_query=rewritten_query, contexts=contexts)

    def score_answer(self, *, question: str, answer: str, contexts: list[dict]) -> float | None:
        return None

    @staticmethod
    def _context_block(contexts: list[dict]) -> str:
        return "\n\n".join(f"[{item['citation']}]\n{item['text']}" for item in contexts)


class FallbackLLMService(BaseLLMService):
    provider = "extractive"

    def answer(self, *, question: str, rewritten_query: str, contexts: list[dict]) -> str:
        lines = ["No LLM provider is configured, so here are the most relevant passages verbatim:"]
        for item in contexts[:3]:
            lines.append(f"- [{item['citation']}] {item['text'][:420].strip()}")
        return "\n".join(lines)


class ChatLLMService(BaseLLMService):
    """Shared prompting, budget guards and error mapping. Subclasses implement two primitives."""

    provider = "chat"

    # ---- provider primitives -------------------------------------------------
    def _complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError

    def _stream(self, *, system: str, user: str, max_tokens: int, temperature: float) -> Iterator[str]:
        raise NotImplementedError

    # ---- pipeline ------------------------------------------------------------
    def rewrite_query(self, question: str) -> str:
        self._guard_budget(input_tokens=estimate_tokens(question), output_tokens=80)
        try:
            with self._slot():
                text = self._complete(system=REWRITE_SYSTEM, user=question, max_tokens=80, temperature=0)
        except HTTPException:
            raise
        except Exception as exc:
            if self._is_throttle(exc):
                self._raise_provider_error(exc)
            logger.warning("%s rewrite failed (%s); using the original question", self.provider, type(exc).__name__)
            return question
        return (text or question).strip()

    def _answer_prompt(self, question: str, rewritten_query: str, contexts: list[dict]) -> tuple[str, int]:
        user = (
            f"Original question:\n{question}\n\nRewritten retrieval query:\n{rewritten_query}\n\n"
            f"Context passages:\n{self._context_block(contexts)}"
        )
        return user, estimate_tokens(user)

    def answer(self, *, question: str, rewritten_query: str, contexts: list[dict]) -> str:
        user, prompt_tokens = self._answer_prompt(question, rewritten_query, contexts)
        self._guard_budget(input_tokens=prompt_tokens, output_tokens=settings.max_generation_tokens)
        try:
            with self._slot():
                text = self._complete(
                    system=ANSWER_SYSTEM, user=user, max_tokens=settings.max_generation_tokens, temperature=0.1
                )
        except HTTPException:
            raise
        except Exception as exc:
            if self._is_throttle(exc):
                self._raise_provider_error(exc)
            return self._degraded_answer(exc, question, rewritten_query, contexts)
        return (text or "").strip()

    def stream_answer(self, *, question: str, rewritten_query: str, contexts: list[dict]) -> Generator[str, None, None]:
        user, prompt_tokens = self._answer_prompt(question, rewritten_query, contexts)
        self._guard_budget(input_tokens=prompt_tokens, output_tokens=settings.max_generation_tokens)
        produced = False
        try:
            with self._slot():
                for piece in self._stream(
                    system=ANSWER_SYSTEM, user=user, max_tokens=settings.max_generation_tokens, temperature=0.1
                ):
                    produced = True
                    yield piece
        except HTTPException:
            raise
        except Exception as exc:
            if produced or self._is_throttle(exc):
                self._raise_provider_error(exc)
            yield self._degraded_answer(exc, question, rewritten_query, contexts)

    def score_answer(self, *, question: str, answer: str, contexts: list[dict]) -> float | None:
        if not settings.groq_answer_scoring_enabled:
            return None
        user = f"Question: {question}\n\nAnswer: {answer}\n\nContext:\n{self._context_block(contexts)}"
        self._guard_budget(input_tokens=estimate_tokens(user), output_tokens=60)
        try:
            with self._slot():
                content = self._complete(system=SCORE_SYSTEM, user=user, max_tokens=60, temperature=0)
        except Exception:
            return None
        try:
            start, end = content.find("{"), content.rfind("}")
            value = json.loads(content[start : end + 1]).get("score")
            return round(float(value), 4) if value is not None else None
        except Exception:
            return None

    # ---- guards ----------------------------------------------------------------
    def _guard_budget(self, *, input_tokens: int, output_tokens: int) -> None:
        total = max(input_tokens + output_tokens, 1)
        checks = [
            ("llm:requests:minute", settings.groq_requests_per_minute, 60, 1, "LLM request budget exhausted for this minute. Please retry shortly."),
            ("llm:requests:day", settings.groq_requests_per_day, 86400, 1, "LLM daily request budget exhausted for this demo."),
            ("llm:tokens:minute", settings.groq_tokens_per_minute, 60, total, "LLM token budget exhausted for this minute. Please retry shortly."),
            ("llm:tokens:day", settings.groq_tokens_per_day, 86400, total, "LLM daily token budget exhausted for this demo."),
        ]
        for key, limit, window, amount, message in checks:
            result = rate_limit_store.hit(key=key, limit=limit, window_seconds=window, amount=amount)
            if not result["allowed"]:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=message)

    @staticmethod
    def _slot():
        class _Slot:
            def __enter__(self_inner):
                if not provider_semaphore.acquire(timeout=5):
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="LLM concurrency budget exhausted. Please retry shortly.",
                    )
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                provider_semaphore.release()

        return _Slot()

    @staticmethod
    def _is_throttle(exc: Exception) -> bool:
        message = str(exc).lower()
        return "429" in message or "rate limit" in message or "throttl" in message

    def _degraded_answer(self, exc: Exception, question: str, rewritten_query: str, contexts: list[dict]) -> str:
        """Provider unavailable (e.g. model access not yet granted): still return cited evidence."""
        logger.warning("%s answer failed (%s); serving extractive answer", self.provider, type(exc).__name__)
        return FallbackLLMService().answer(question=question, rewritten_query=rewritten_query, contexts=contexts).replace(
            "No LLM provider is configured", f"The {self.provider} model is unavailable right now", 1
        )

    def _raise_provider_error(self, exc: Exception) -> None:
        logger.warning("%s provider error: %s", self.provider, exc)
        if self._is_throttle(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"The {self.provider} model is rate limiting this demo right now. Please retry shortly.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The {self.provider} model is temporarily unavailable for this demo.",
        ) from exc


class GroqLLMService(ChatLLMService):
    provider = "groq"

    def __init__(self, api_key: str, model: str) -> None:
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model

    def _messages(self, system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            messages=self._messages(system, user),
        )
        return response.choices[0].message.content or ""

    def _stream(self, *, system: str, user: str, max_tokens: int, temperature: float) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
            messages=self._messages(system, user),
        )
        for item in stream:
            if item.choices and item.choices[0].delta.content:
                yield item.choices[0].delta.content


class BedrockLLMService(ChatLLMService):
    provider = "bedrock"

    def __init__(self, model_id: str, region: str | None) -> None:
        import boto3
        from botocore.config import Config

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}, read_timeout=60),
        )
        self.model_id = model_id

    def _request(self, system: str, user: str, max_tokens: int, temperature: float) -> dict:
        return {
            "modelId": self.model_id,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }

    def _complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        response = self.client.converse(**self._request(system, user, max_tokens, temperature))
        blocks = response["output"]["message"]["content"]
        return "".join(block.get("text", "") for block in blocks)

    def _stream(self, *, system: str, user: str, max_tokens: int, temperature: float) -> Iterator[str]:
        response = self.client.converse_stream(**self._request(system, user, max_tokens, temperature))
        for event in response["stream"]:
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if delta.get("text"):
                yield delta["text"]


def _build_service() -> BaseLLMService:
    try:
        if settings.llm_provider == "bedrock":
            return BedrockLLMService(model_id=settings.bedrock_model_id, region=settings.aws_region)
        if settings.llm_provider == "groq" and settings.groq_api_key:
            return GroqLLMService(api_key=settings.groq_api_key, model=settings.groq_model)
    except Exception:  # pragma: no cover - depends on environment
        logger.exception("Configured LLM provider %s unavailable; using extractive fallback", settings.llm_provider)
    return FallbackLLMService()


llm_service = _build_service()
