import yaml
from pathlib import Path

import model_form
from translator_anthropic import ANTHROPIC_PROVIDER_PRESET


def test_state_openai():
    st = model_form.compute_field_state("auto", "https://api.deepseek.com/v1", "x")
    assert not st["is_anthropic"] and st["show_no_system_role"]
    assert st["sampling_enabled"] and not st["show_effort"]
    assert "anthropic" not in st["thinking_styles"]


def test_state_anthropic_auto_and_fallback():
    st = model_form.compute_field_state("auto", "https://api.anthropic.com", "claude-opus-5-5")
    assert st["is_anthropic"] and st["fallback_enabled"] and st["show_effort"]
    assert not st["sampling_enabled"] and not st["show_no_system_role"]
    assert "anthropic" in st["thinking_styles"]
    st = model_form.compute_field_state("auto", "https://api.anthropic.com", "claude-haiku-5-5")
    assert not st["fallback_enabled"]
    st = model_form.compute_field_state("anthropic", "https://gw.example.com/anthropic", "claude-opus-5-5")
    assert st["is_anthropic"] and not st["fallback_enabled"]


def test_explicit_openai_overrides_auto_detection():
    st = model_form.compute_field_state("openai", "https://api.anthropic.com", "m")
    assert not st["is_anthropic"]


def test_fill_prices():
    assert model_form.fill_prices_if_empty("claude-sonnet-5-5", 0, 0) == (2.0, 10.0)
    assert model_form.fill_prices_if_empty("claude-sonnet-5-5", 5, 0) == (5, 10.0)
    assert model_form.fill_prices_if_empty("gpt-x", 0, 0) == (0, 0)


def _form(**kw):
    f = dict(name=" n ", api_base="https://api.anthropic.com", api_key="k",
             model="claude-opus-5-5", proxy="none", api_format="auto",
             thinking_style="auto", streaming=True, context_turns=0,
             input_price=0, output_price=0)
    f.update(kw)
    return f


def test_build_anthropic():
    d = model_form.build_model_dict(_form(
        api_format="anthropic", effort="high", anthropic_fallbacks=False,
        no_system_role=True, thinking_style="anthropic",
        overrides={"temperature": 0.5, "max_tokens": 512}))
    assert d["name"] == "n" and d["api_format"] == "anthropic"
    assert d["overrides"] == {"max_tokens": 512, "effort": "high"}
    assert d["anthropic_fallbacks"] is False
    assert "no_system_role" not in d and d["thinking_style"] == "anthropic"


def test_build_openai_drops_anthropic_fields():
    d = model_form.build_model_dict(_form(
        api_base="https://api.openai.com/v1", effort="high",
        thinking_style="anthropic", no_system_role=True,
        overrides={"temperature": 0.5}, anthropic_fallbacks=False))
    assert d["overrides"] == {"temperature": 0.5}
    assert "api_format" not in d and "thinking_style" not in d
    assert "anthropic_fallbacks" not in d and d["no_system_role"] is True


def test_preset_roundtrip():
    p = model_form.preset_values()
    assert p == ANTHROPIC_PROVIDER_PRESET
    d = model_form.build_model_dict(_form(**{k: p[k] for k in
        ("name", "api_base", "model", "api_format", "input_price", "output_price")}))
    assert d["api_format"] == "anthropic" and d["input_price"] > 0


def _keys(d, prefix=""):
    out = set()
    for k, v in d.items():
        out.add(prefix + str(k))
        if isinstance(v, dict):
            out |= _keys(v, prefix + str(k) + ".")
    return out


def test_i18n_key_parity():
    base = Path(__file__).parent.parent / "i18n"
    en = yaml.safe_load((base / "en.yaml").read_text("utf-8"))
    zh = yaml.safe_load((base / "zh.yaml").read_text("utf-8"))
    assert _keys(en) == _keys(zh)
