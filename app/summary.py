from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT_SECONDS

LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = """You are a digital-forensics analyst. Using ONLY the evidence provided, write plain-language markdown with exactly these ## headings, in this order: "Verdict", "Key reasons", "What's uncertain", "Next steps". Under each heading use short bullets. State the verdict and confidence first. Explain technical terms in parentheses. Reference specific evidence and video timestamps when provided. Never claim certainty, never invent evidence, and do not add other headings."""


@dataclass(frozen=True)
class SummaryResult:
    markdown: str
    is_fallback: bool


def _fallback(evidence: dict[str, Any]) -> str:
    lines = [
        "## Verdict",
        f"- **{evidence.get('verdict', 'Inconclusive')}** with {evidence.get('confidence', 'Low')} confidence.",
        f"- Fake-likelihood score: {evidence.get('fake_score', 50)}/100.",
        "",
        "## Key reasons",
    ]
    for item in evidence.get("evidence", [])[:5]:
        lines.append(f"- **{item.get('signal', 'Signal')}:** {item.get('finding', 'No finding provided')}")
    lines += [
        "",
        "## What's uncertain",
        "- This is a probabilistic screening result, not proof. Compression, missing faces, metadata stripping, and detector disagreement can affect it.",
        *[f"- {item}" for item in evidence.get("limitations", [])],
        "",
        "## Next steps",
        "- Preserve the original file and compare it with a trusted source or capture history.",
    ]
    return "\n".join(lines)


def deterministic_fallback(evidence: dict[str, Any]) -> str:
    return _fallback(evidence)


def _log_failure(exc: Exception) -> None:
    name = type(exc).__name__
    if name == "AuthenticationError":
        reason = "authentication failed; check GROQ_API_KEY"
    elif name == "RateLimitError":
        reason = "rate limited by Groq"
    elif name in {"APITimeoutError", "TimeoutError"}:
        reason = "request timed out"
    elif name in {"APIConnectionError", "ConnectError"}:
        reason = "network connection failed"
    elif name in {"BadRequestError", "NotFoundError"}:
        reason = "request rejected; check GROQ_MODEL and payload"
    elif isinstance(exc, ValueError):
        reason = "provider returned an empty or invalid explanation"
    else:
        reason = "provider request failed"
    LOGGER.warning("Groq summary unavailable (%s): %s; using deterministic fallback", reason, name)


def _request_summary(evidence: dict[str, Any]) -> str:
    client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS, max_retries=2)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(evidence, default=str)},
        ],
    )
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    text = getattr(message, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Groq returned an empty explanation")
    return text.strip()


async def generate_summary(evidence: dict[str, Any]) -> SummaryResult:
    if not GROQ_API_KEY:
        LOGGER.info("GROQ_API_KEY is not configured; using deterministic fallback")
        return SummaryResult(_fallback(evidence), True)
    try:
        text = await asyncio.to_thread(_request_summary, evidence)
        return SummaryResult(text, False)
    except Exception as exc:
        _log_failure(exc)
        return SummaryResult(_fallback(evidence), True)
