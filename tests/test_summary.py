import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app import summary


EVIDENCE = {
    "verdict": "Inconclusive",
    "confidence": "Low",
    "fake_score": 50,
    "evidence": [{"signal": "ensemble", "finding": "Models disagree", "score": 0.5}],
    "limitations": ["Compression may affect the result."],
}


class AuthenticationError(Exception):
    pass


class RateLimitError(Exception):
    pass


class APITimeoutError(Exception):
    pass


def _client_with(content):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    return client


def test_successful_groq_summary():
    with patch.object(summary, "GROQ_API_KEY", "test-key"), patch.object(
        summary, "Groq", return_value=_client_with("## Verdict\n- Looks real.")
    ):
        result = asyncio.run(summary.generate_summary(EVIDENCE))
    assert result.markdown == "## Verdict\n- Looks real."
    assert result.is_fallback is False


def test_missing_key_uses_deterministic_fallback():
    with patch.object(summary, "GROQ_API_KEY", ""):
        result = asyncio.run(summary.generate_summary(EVIDENCE))
    assert result.is_fallback is True
    assert "## Verdict" in result.markdown


@pytest.mark.parametrize("error", [AuthenticationError(), RateLimitError(), APITimeoutError()])
def test_provider_failures_use_fallback(error):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(error))
        )
    )
    with patch.object(summary, "GROQ_API_KEY", "test-key"), patch.object(
        summary, "Groq", return_value=client
    ):
        result = asyncio.run(summary.generate_summary(EVIDENCE))
    assert result.is_fallback is True
    assert "## Next steps" in result.markdown


def test_empty_response_uses_fallback():
    with patch.object(summary, "GROQ_API_KEY", "test-key"), patch.object(
        summary, "Groq", return_value=_client_with("")
    ):
        result = asyncio.run(summary.generate_summary(EVIDENCE))
    assert result.is_fallback is True
