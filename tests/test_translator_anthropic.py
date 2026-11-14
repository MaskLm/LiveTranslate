from types import SimpleNamespace as NS

import pytest

import translator_anthropic as tla
from translator import RepetitionError, Translator

anthropic = pytest.importorskip("anthropic")


# --------------------------------------------------------------- resolver
@pytest.mark.parametrize(
    "fmt,base,expected",
    [
        ("auto", "https://api.anthropic.com", "anthropic"),
        (None, "https://api.anthropic.com/v1", "anthropic"),
        ("auto", "https://api.deepseek.com/anthropic", "anthropic"),
        ("auto", "https://gw.example.com/anthropic/", "anthropic"),
        ("auto", "https://gw.example.com/anthropic/v1", "anthropic"),
        ("auto", "https://api.openai.com/v1", "openai"),
        ("auto", "http://localhost:1234/v1", "openai"),
        ("auto", "https://notapi.anthropic.com.evil.com/v1", "openai"),
        ("openai", "https://api.anthropic.com", "openai"),
        ("anthropic", "https://x.example.com/v1", "anthropic"),
        ("ANTHROPIC", "https://x.example.com/v1", "anthropic"),
        ("bogus", "https://api.openai.com/v1", "openai"),
    ],
)
def test_resolve_api_format(fmt, base, expected):
    assert tla.resolve_api_format(fmt, base) == expected


def test_normalize_base_url():
    assert tla.normalize_anthropic_base_url("https://api.anthropic.com/v1/") == "https://api.anthropic.com"
    assert tla.normalize_anthropic_base_url("https://gw.example.com/anthropic") == "https://gw.example.com/anthropic"


def test_preset_and_prices():
    p = tla.ANTHROPIC_PROVIDER_PRESET
    assert p["name"] == "Anthropic Claude" and p["model"] == "claude-opus-5-5"
    assert p["models"] == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    assert tla.anthropic_model_prices("claude-sonnet-5-5") == (2.0, 10.0)
    assert tla.anthropic_model_prices("claude-haiku-5-5") == (0.10, 0.50)
    assert tla.anthropic_model_prices("gpt-4o") is None


# ------------------------------------------------------- client creation
@pytest.mark.parametrize("proxy", ["none", "system", "http://127.0.0.1:7890"])
def test_make_client(proxy):
    c = tla.make_anthropic_client("https://api.anthropic.com/v1", "k", proxy, timeout=10)
    assert str(c.base_url).rstrip("/") == "https://api.anthropic.com"


# --------------------------------------------------------- request kwargs
def _req(**kw):
    base = dict(
        model="claude-opus-5-5",
        system_prompt="SYS",
        text="hello",
        api_base="https://api.anthropic.com",
    )
    base.update(kw)
    return tla.build_request(**base)


def test_request_basics():
    kw, beta = _req(
        overrides={"temperature": 0.3, "top_p": 0.9, "seed": 1,
                   "frequency_penalty": 0.1, "presence_penalty": 0.1},
        history=[("a", "A"), ("b", "B")],
    )
    for k in ("temperature", "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty"):
        assert k not in kw
    assert kw["system"] == "SYS"
    assert kw["max_tokens"] == tla.ANTHROPIC_DEFAULT_MAX_TOKENS
    assert kw["output_config"] == {"effort": "low"}
    assert "thinking" not in kw
    roles = [m["role"] for m in kw["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in kw["messages"]] == ["a", "A", "b", "B", "hello"]
    assert all(m["role"] != "system" for m in kw["messages"])
    assert beta is True


def test_max_tokens_and_effort_override():
    kw, _ = _req(overrides={"max_tokens": 77, "effort": "medium"})
    assert kw["max_tokens"] == 77
    assert kw["output_config"]["effort"] == "medium"


def test_json_schema_and_extra_body():
    kw, _ = _req(
        json_response=True,
        extra_body={"temperature": 1, "metadata": {"user_id": "u"},
                    "output_config": {"effort": "high"}},
    )
    assert kw["output_config"]["effort"] == "high"
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"] == {"t": {"type": "string"}}
    assert kw["extra_body"] == {"metadata": {"user_id": "u"}}


@pytest.mark.parametrize(
    "model,style,expected",
    [
        ("claude-sonnet-5-5", "anthropic", {"type": "between_tools"}),
        ("claude-sonnet-5-5", "auto", None),
        ("claude-opus-5-5", "anthropic", None),
        ("claude-haiku-5-5", "anthropic", None),
    ],
)
def test_thinking_mapping(model, style, expected):
    kw, _ = _req(model=model, thinking_style=style)
    assert kw.get("thinking") == expected
    assert kw["output_config"]["effort"] == "low"
    assert kw.get("thinking", {}).get("type") != "disabled"


def test_between_tools_skipped_for_xhigh_effort():
    kw, _ = _req(model="claude-sonnet-5-5", thinking_style="anthropic",
                 overrides={"effort": "xhigh"})
    assert "thinking" not in kw


def test_gateway_models_get_no_effort_by_default():
    kw, beta = _req(model="glm-4.6", api_base="https://gw.example.com/anthropic")
    assert "output_config" not in kw and beta is False
    kw, _ = _req(model="glm-4.6", api_base="https://gw.example.com/anthropic",
                 overrides={"effort": "low"})
    assert kw["output_config"] == {"effort": "low"}


@pytest.mark.parametrize(
    "model,base,fallbacks,expected",
    [
        ("claude-opus-5-5", "https://api.anthropic.com", True, True),
        ("claude-sonnet-5-5", "https://api.anthropic.com", True, True),
        ("claude-fable-5-2", "https://api.anthropic.com", True, True),
        ("claude-haiku-5-5", "https://api.anthropic.com", True, False),
        ("claude-opus-5-5", "https://api.anthropic.com", False, False),
        ("claude-opus-5-5", "https://gw.example.com/anthropic", True, False),
    ],
)
def test_fallbacks_first_party_only(model, base, fallbacks, expected):
    kw, beta = _req(model=model, api_base=base, fallbacks=fallbacks)
    assert beta is expected
    if expected:
        assert kw["betas"] == ["server-side-fallback-2026-07-01"]
        assert kw["fallbacks"] == "default"
    else:
        assert "betas" not in kw and "fallbacks" not in kw


# --------------------------------------------------- fake client / streaming
class FakeStream:
    def __init__(self, pieces, final):
        self._pieces, self._final = pieces, final
        self.text_stream = iter(pieces)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._final


def _msg(text="", stop="end_turn", usage=(10, 5), content=None, details=None):
    return NS(
        content=content if content is not None else [NS(type="text", text=text)],
        stop_reason=stop,
        stop_details=details,
        usage=NS(input_tokens=usage[0], output_tokens=usage[1],
                 cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


class FakeMessages:
    def __init__(self, pieces=None, final=None):
        self.pieces, self.final, self.calls = pieces or [], final, []

    def create(self, **kw):
        self.calls.append(("create", kw))
        return self.final

    def stream(self, **kw):
        self.calls.append(("stream", kw))
        return FakeStream(self.pieces, self.final)


def _translator(fake, **kw):
    t = Translator(
        "https://api.anthropic.com", "k", "claude-opus-5-5",
        streaming=kw.pop("streaming", True), **kw,
    )
    t._client = NS(messages=fake, beta=NS(messages=fake))
    return t


def test_streaming_accumulates_and_counts_usage():
    fake = FakeMessages(["Hel", "lo ", "world"], _msg("Hello world", usage=(42, 7)))
    t = _translator(fake)
    out = list(t.translate_iter("hi", "en"))
    assert out[-1] == "Hello world"
    assert out[:3] == ["Hel", "Hello ", "Hello world"]
    assert t.last_usage == (42, 7)
    kind, kw = fake.calls[0]
    assert kind == "stream" and kw["fallbacks"] == "default"
    assert "temperature" not in kw


def test_streaming_ignores_thinking_blocks_and_updates_history():
    final = _msg(content=[NS(type="thinking", thinking=""), NS(type="text", text="你好")])
    t = _translator(FakeMessages(["你好"], final))
    t.set_context_turns(2)
    assert t.translate("hi", "en") == "你好"
    assert t._history == [("hi", "你好")]


def test_history_sent_as_turns():
    fake = FakeMessages([], _msg("x"))
    t = _translator(fake, streaming=False)
    t.set_context_turns(2)
    t._history = [("a", "A")]
    t.translate("b", "en")
    kind, kw = fake.calls[0]
    assert kind == "create"
    assert kw["messages"] == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "b"},
    ]


def test_sync_json_response_and_usage():
    fake = FakeMessages(final=_msg('{"t": "bonjour"}', usage=(3, 4)))
    t = _translator(fake, streaming=False, json_response=True)
    assert t.translate("hi", "en") == "bonjour"
    assert t.last_usage == (3, 4)


def test_refusal_raises_and_is_not_added_to_history():
    details = NS(category="cyber", explanation=None)
    fake = FakeMessages(["par"], _msg("", stop="refusal", details=details))
    t = _translator(fake)
    t.set_context_turns(2)
    with pytest.raises(tla.TranslationRefusedError, match="cyber"):
        list(t.translate_iter("hi", "en"))
    assert t._history == []
    t2 = _translator(FakeMessages(final=_msg("", stop="refusal")), streaming=False)
    with pytest.raises(tla.TranslationRefusedError):
        t2.translate("hi", "en")


def test_max_tokens_warns(caplog):
    t = _translator(FakeMessages(final=_msg("partial", stop="max_tokens")), streaming=False)
    with caplog.at_level("WARNING", logger="LiveTranslate.TL"):
        assert t.translate("hi", "en") == "partial"
    assert "max_tokens" in caplog.text


def test_repetition_detected_on_anthropic_path():
    rep = "abcdefgh" * 8
    t = _translator(FakeMessages([rep], _msg(rep)))
    with pytest.raises(RepetitionError):
        list(t.translate_iter("hi", "en"))


def test_with_target_language_keeps_format():
    t = _translator(FakeMessages(final=_msg("x")), streaming=False)
    t2 = t.with_target_language("ja")
    assert t2._api_format == "anthropic" and t2.translate("a", "en") == "x"


def test_openai_path_untouched():
    t = Translator("https://api.openai.com/v1", "k", "gpt-4o")
    assert t._api_format == "openai"
    kw = t._build_request_kwargs("sys", "hi")
    assert kw["temperature"] == 0.3 and kw["messages"][0]["role"] == "system"


# ------------------------------------------------------------ error mapping
def _resp(status, headers=None):
    import httpx2

    return httpx2.Response(
        status, headers=headers or {},
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


def test_error_mapping_order():
    m = tla.map_anthropic_error
    auth = m(anthropic.AuthenticationError("bad key", response=_resp(401), body=None))
    assert isinstance(auth, tla.TranslationAPIError) and "401" in str(auth)
    rl = m(anthropic.RateLimitError("slow", response=_resp(429, {"retry-after": "7"}), body=None))
    assert "rate limit" in str(rl) and "7" in str(rl)
    st = m(anthropic.InternalServerError("boom", response=_resp(500), body=None))
    assert isinstance(st, tla.TranslationAPIError) and "500" in str(st)
    req = _resp(500).request
    assert isinstance(m(anthropic.APITimeoutError(request=req)), TimeoutError)
    assert isinstance(m(anthropic.APIConnectionError(request=req)), ConnectionError)
    plain = ValueError("x")
    assert m(plain) is plain


def test_sdk_error_is_mapped_in_translate():
    class Boom(FakeMessages):
        def create(self, **kw):
            raise anthropic.AuthenticationError("bad", response=_resp(401), body=None)

    t = _translator(Boom(), streaming=False)
    with pytest.raises(tla.TranslationAPIError):
        t.translate("hi", "en")
