"""Pure (Qt-free) helpers behind ModelEditDialog so they can be unit-tested."""

from translator_anthropic import (
    ANTHROPIC_DEFAULT_EFFORT,
    ANTHROPIC_EFFORTS,
    ANTHROPIC_PROVIDER_PRESET,
    FALLBACK_MODELS,
    SAMPLING_KEYS,
    anthropic_model_prices,
    is_first_party_anthropic,
    resolve_api_format,
)

BASE_THINKING_STYLES = ("auto", "deepseek", "qwen", "vllm", "openai", "off")
ANTHROPIC_THINKING_OPTION = "anthropic"
# Override keys the dialog edits with spin boxes (effort is handled separately)
SAMPLING_OVERRIDE_KEYS = tuple(k for k in SAMPLING_KEYS if k != "top_k")


def compute_field_state(api_format, api_base, model) -> dict:
    """Which dialog fields are visible/enabled for (api_format, api_base, model)."""
    resolved = resolve_api_format(api_format, api_base)
    anth = resolved == "anthropic"
    fallback_ok = (
        anth
        and is_first_party_anthropic(api_base)
        and str(model or "").strip().lower() in FALLBACK_MODELS
    )
    styles = list(BASE_THINKING_STYLES)
    if anth:
        styles.append(ANTHROPIC_THINKING_OPTION)
    return {
        "resolved_format": resolved,
        "is_anthropic": anth,
        "model_suggestions": True if anth else False,
        "show_effort": anth,
        "show_fallback": anth,
        "fallback_enabled": fallback_ok,
        "show_no_system_role": not anth,
        "sampling_enabled": not anth,
        "thinking_styles": styles,
    }


def fill_prices_if_empty(model, input_price, output_price):
    """Return (input, output); a known Claude model fills zero/empty prices."""
    known = anthropic_model_prices(model)
    inp, out = input_price or 0, output_price or 0
    if known:
        if not inp:
            inp = known[0]
        if not out:
            out = known[1]
    return inp, out


def preset_values() -> dict:
    return dict(ANTHROPIC_PROVIDER_PRESET)


def build_model_dict(form: dict) -> dict:
    """Build the saved model dict from raw form values.

    Keys: name, api_base, api_key, model, proxy, api_format, no_system_role,
    thinking_style, streaming, json_response, context_turns, input_price,
    output_price, overrides (dict of *enabled* overrides), effort (str|None,
    enabled effort override), anthropic_fallbacks (bool), extra_body.
    """
    api_format = form.get("api_format", "auto") or "auto"
    state = compute_field_state(api_format, form.get("api_base"), form.get("model"))
    anth = state["is_anthropic"]
    result = {
        "name": str(form.get("name", "")).strip(),
        "api_base": str(form.get("api_base", "")).strip(),
        "api_key": str(form.get("api_key", "")).strip(),
        "model": str(form.get("model", "")).strip(),
        "proxy": form.get("proxy", "none") or "none",
    }
    if api_format != "auto":
        result["api_format"] = api_format
    if anth:
        if form.get("anthropic_fallbacks", True) is False:
            result["anthropic_fallbacks"] = False
    elif form.get("no_system_role"):
        result["no_system_role"] = True
    style = form.get("thinking_style", "auto")
    if style not in state["thinking_styles"]:
        style = "auto"
    if style != "auto":
        result["thinking_style"] = style
    if form.get("streaming", True) is False:
        result["streaming"] = False
    if form.get("json_response"):
        result["json_response"] = True
    if form.get("context_turns", 0) > 0:
        result["context_turns"] = form["context_turns"]
    if form.get("input_price", 0) > 0:
        result["input_price"] = form["input_price"]
    if form.get("output_price", 0) > 0:
        result["output_price"] = form["output_price"]

    overrides = {}
    for key, val in (form.get("overrides") or {}).items():
        if anth and key in SAMPLING_OVERRIDE_KEYS:
            continue
        overrides[key] = val
    effort = form.get("effort")
    if anth and effort in ANTHROPIC_EFFORTS:
        overrides["effort"] = effort
    if overrides:
        result["overrides"] = overrides
    if form.get("extra_body"):
        result["extra_body"] = form["extra_body"]
    return result


DEFAULT_EFFORT = ANTHROPIC_DEFAULT_EFFORT
