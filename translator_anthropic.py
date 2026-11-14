"""Native Anthropic (Claude) Messages API support for the translator.

Everything here is pure / SDK-light so it can be unit-tested without network.
The ``anthropic`` package is imported lazily so an install that has not yet
picked up the new requirement keeps working for every other provider.
"""

import logging
import re
from urllib.parse import urlparse

log = logging.getLogger("LiveTranslate.TL")

# ---------------------------------------------------------------------------
# Data hooks for the UI (provider preset, model list, prices, option lists)
# ---------------------------------------------------------------------------
API_FORMATS = ("auto", "openai", "anthropic")

ANTHROPIC_API_BASE = "https://api.anthropic.com"
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
ANTHROPIC_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]

# USD per 1M tokens: (input, output)
ANTHROPIC_MODEL_PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

ANTHROPIC_PROVIDER_PRESET = {
    "name": "Anthropic Claude",
    "api_base": ANTHROPIC_API_BASE,
    "model": ANTHROPIC_DEFAULT_MODEL,
    "models": list(ANTHROPIC_MODELS),
    "api_format": "anthropic",
    "input_price": ANTHROPIC_MODEL_PRICES[ANTHROPIC_DEFAULT_MODEL][0],
    "output_price": ANTHROPIC_MODEL_PRICES[ANTHROPIC_DEFAULT_MODEL][1],
}

# thinking_style value meaning "no thinking" on the Anthropic path
ANTHROPIC_THINKING_STYLE = "anthropic"
# Values accepted by overrides["effort"]
ANTHROPIC_EFFORTS = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_DEFAULT_EFFORT = "low"
# Thinking counts toward max_tokens, so do not inherit the global 256 default
ANTHROPIC_DEFAULT_MAX_TOKENS = 1024

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Server-side refusal fallback is offered for these models only (Haiku 5.5 has none)
FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2")
_FIRST_PARTY_HOST = "api.anthropic.com"

# Rejected by the current API version on every model
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)

_JSON_SCHEMA = {
    "type": "object",
    "properties": {"t": {"type": "string"}},
    "required": ["t"],
    "additionalProperties": False,
}

# Model ids known to accept output_config.effort
_EFFORT_MODEL_RE = re.compile(
    r"^claude-(opus-(4-[5-9]|5)|sonnet-(4-[6-9]|5)|haiku-5|fable|mythos)"
)


class TranslationAPIError(RuntimeError):
    """Provider returned an error (auth, rate limit, HTTP status)."""


class TranslationRefusedError(TranslationAPIError):
    """The model declined the request (stop_reason == "refusal")."""


# ---------------------------------------------------------------------------
# api_format resolution
# ---------------------------------------------------------------------------
def _split_base(api_base):
    base = str(api_base or "").strip()
    parsed = urlparse(base if "://" in base else "//" + base)
    return (parsed.hostname or "").lower(), (parsed.path or "").rstrip("/")


def is_first_party_anthropic(api_base) -> bool:
    return _split_base(api_base)[0] == _FIRST_PARTY_HOST


def resolve_api_format(api_format, api_base) -> str:
    """Resolve a per-model api_format setting to "openai" or "anthropic".

    "auto" picks anthropic for api.anthropic.com and for Anthropic-compatible
    gateways whose base URL path ends in /anthropic (optionally /anthropic/v1).
    """
    fmt = str(api_format or "auto").strip().lower()
    if fmt in ("openai", "anthropic"):
        return fmt
    host, path = _split_base(api_base)
    if host == _FIRST_PARTY_HOST:
        return "anthropic"
    path = path.lower()
    if path.endswith("/v1"):
        path = path[:-3]
    if path.endswith("/anthropic"):
        return "anthropic"
    return "openai"


def normalize_anthropic_base_url(api_base) -> str:
    """The SDK appends /v1/messages itself; drop a pasted OpenAI-style /v1."""
    base = str(api_base or "").strip().rstrip("/")
    if base.lower().endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base or ANTHROPIC_API_BASE


def anthropic_model_prices(model):
    """(input, output) USD per 1M tokens for a known Claude model, else None."""
    return ANTHROPIC_MODEL_PRICES.get(str(model or "").strip().lower())


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def make_anthropic_client(api_base, api_key, proxy="none", timeout=None):
    """Proxy-aware Anthropic client, mirroring translator.make_openai_client.

    anthropic 1.x+ runs on httpx2, so the HTTP client must come from the SDK
    (DefaultHttpxClient), never from the ``httpx`` package.
    """
    import anthropic

    kwargs = {
        "api_key": api_key,
        "base_url": normalize_anthropic_base_url(api_base),
    }
    if timeout is not None:
        kwargs["timeout"] = anthropic.Timeout(timeout, connect=5.0)
    if proxy == "system":
        pass
    elif proxy in ("none", "", None):
        kwargs["http_client"] = anthropic.DefaultHttpxClient(trust_env=False)
    else:
        kwargs["http_client"] = anthropic.DefaultHttpxClient(proxy=proxy)
    return anthropic.Anthropic(**kwargs)


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------
def supports_effort(model) -> bool:
    return bool(_EFFORT_MODEL_RE.match(str(model or "").lower()))


def strip_sampling(params):
    """Return (params without sampling keys, [dropped key names])."""
    kept, dropped = {}, []
    for k, v in (params or {}).items():
        if k in SAMPLING_KEYS:
            dropped.append(k)
        else:
            kept[k] = v
    return kept, dropped


def build_messages(history, text):
    """history: iterable of (source, translation). Alternating user/assistant."""
    msgs = []
    for src, tgt in history or ():
        msgs.append({"role": "user", "content": src})
        msgs.append({"role": "assistant", "content": tgt})
    msgs.append({"role": "user", "content": text})
    return msgs


def build_request(
    *,
    model,
    system_prompt,
    text,
    history=(),
    api_base="",
    max_tokens=None,
    overrides=None,
    extra_body=None,
    thinking_style=None,
    json_response=False,
    fallbacks=True,
):
    """Assemble kwargs for ``client.messages.create`` / ``.stream``.

    Returns ``(kwargs, use_beta)``; when ``use_beta`` is True the call must go
    through ``client.beta.messages`` (kwargs already carry betas/fallbacks).
    Never emits temperature/top_p/top_k or assistant prefill.
    """
    overrides = dict(overrides or {})
    model_id = str(model or "")
    lowered = model_id.lower()

    effort_set = "effort" in overrides
    effort = overrides.pop("effort", None)
    ov_max = overrides.pop("max_tokens", None)
    # remaining overrides are all OpenAI-only knobs (sampling etc.)

    kwargs = {
        "model": model_id,
        "max_tokens": int(ov_max if ov_max is not None else
                          (max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS)),
        "system": system_prompt,
        "messages": build_messages(history, text),
    }

    output_config = {}
    if effort_set:
        if effort not in (None, "", "none", "off", False):
            output_config["effort"] = str(effort)
    elif supports_effort(lowered):
        output_config["effort"] = ANTHROPIC_DEFAULT_EFFORT

    thinking = None
    if thinking_style == ANTHROPIC_THINKING_STYLE:
        # "No thinking": only Sonnet 5.5 can go below adaptive (between_tools);
        # Opus 5.5 / Haiku 5.5 reject disabled and budgets, so stay at low effort.
        if lowered == "claude-sonnet-5-5":
            if output_config.get("effort", "low") in ("low", "medium", "high"):
                thinking = {"type": "between_tools"}
            else:
                log.info("Anthropic: between_tools needs effort <= high; keeping adaptive")
    elif thinking_style == "deepseek" and not lowered.startswith("claude-"):
        # Anthropic-compatible gateways (DeepSeek/GLM/Kimi) take the nested shape
        thinking = {"type": "disabled"}

    body, _ = strip_sampling(extra_body)
    # Lift fields that would otherwise clobber ours when merged into the body
    user_output_config = body.pop("output_config", None)
    if isinstance(user_output_config, dict):
        output_config.update(user_output_config)
    user_thinking = body.pop("thinking", None)
    if user_thinking is not None:
        thinking = user_thinking

    if json_response:
        output_config["format"] = {"type": "json_schema", "schema": _JSON_SCHEMA}
    if output_config:
        kwargs["output_config"] = output_config
    if thinking is not None:
        kwargs["thinking"] = thinking
    if body:
        kwargs["extra_body"] = body

    use_beta = bool(
        fallbacks and lowered in FALLBACK_MODELS and is_first_party_anthropic(api_base)
    )
    if use_beta:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs, use_beta


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------
def extract_text(content) -> str:
    """Concatenate text blocks only (skips thinking/fallback blocks).

    A server-side fallback block marks a model switch, so text produced by the
    declining model before it is discarded.
    """
    parts = []
    for block in content or ():
        btype = getattr(block, "type", None)
        if btype == "fallback":
            parts = []
        elif btype == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def usage_tokens(usage):
    """(input_tokens incl. cache, output_tokens) from a Messages usage object."""
    if usage is None:
        return 0, 0
    inp = (
        (getattr(usage, "input_tokens", 0) or 0)
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
    )
    return inp, getattr(usage, "output_tokens", 0) or 0


def check_stop_reason(message, model=""):
    """Raise on refusal, warn on truncation. Call before trusting the text."""
    reason = getattr(message, "stop_reason", None)
    if reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        log.warning(
            f"Anthropic refusal from {model}: category={category} "
            f"explanation={explanation}"
        )
        raise TranslationRefusedError(
            "Claude declined to translate this segment"
            + (f" (category: {category})" if category else "")
        )
    if reason == "max_tokens":
        log.warning(
            f"Anthropic response from {model} hit max_tokens; output may be "
            "truncated or empty (thinking counts toward max_tokens) - raise "
            "max_tokens in Advanced Parameters or lower effort"
        )


def describe_empty(message) -> str:
    """Explain why a response carried no text (block types, stop reason, usage)."""
    types = [getattr(b, "type", "?") for b in (getattr(message, "content", None) or ())]
    reason = getattr(message, "stop_reason", None)
    out = getattr(getattr(message, "usage", None), "output_tokens", None)
    hint = (
        " - thinking likely consumed max_tokens; raise max_tokens or lower effort"
        if reason == "max_tokens" else ""
    )
    return (
        f"Claude returned no text (stop_reason={reason}, blocks={types}, "
        f"output_tokens={out}){hint}"
    )


def _error_type(exc):
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return err.get("type")
    return None


def map_anthropic_error(exc):
    """Map SDK exceptions to the app's error style; others returned unchanged.

    Most specific first: APITimeoutError is a subclass of APIConnectionError and
    AuthenticationError/RateLimitError are subclasses of APIStatusError.
    """
    try:
        import anthropic
    except ImportError:
        return exc
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        log.warning(f"Anthropic auth error: {exc}")
        return TranslationAPIError(
            f"Anthropic authentication failed ({exc.status_code}): check api_key/permissions"
        )
    if isinstance(exc, anthropic.RateLimitError):
        retry = None
        try:
            retry = exc.response.headers.get("retry-after")
        except Exception:
            pass
        log.warning(f"Anthropic rate limited (retry-after={retry})")
        return TranslationAPIError(
            "Anthropic rate limit exceeded" + (f", retry after {retry}s" if retry else "")
        )
    if isinstance(exc, anthropic.APIStatusError):
        msg = getattr(exc, "message", None) or str(exc)
        log.warning(f"Anthropic API error {exc.status_code}: {msg}")
        etype = _error_type(exc)
        return TranslationAPIError(
            f"Anthropic API error {exc.status_code}"
            + (f" ({etype})" if etype else "")
            + f": {msg}"
        )
    if isinstance(exc, anthropic.APITimeoutError):
        return TimeoutError("Anthropic request timed out")
    if isinstance(exc, anthropic.APIConnectionError):
        return ConnectionError(f"Anthropic connection error: {exc}")
    return exc
