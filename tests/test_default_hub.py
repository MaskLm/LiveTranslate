import model_manager as mm


def test_default_is_huggingface():
    assert mm.DEFAULT_HUB == "hf"
    assert mm.HUB_CHOICES[0] == "hf"


def test_missing_setting_resolves_to_hf():
    assert mm.resolve_hub({}) == "hf"
    assert mm.resolve_hub(None) == "hf"
    assert mm.resolve_hub({"hub": None}) == "hf"
    assert mm.resolve_hub({"hub": "bogus"}) == "hf"


def test_explicit_saved_value_kept():
    assert mm.resolve_hub({"hub": "ms"}) == "ms"
    assert mm.resolve_hub({"hub": "hf"}) == "hf"


def test_chinese_ui_language_does_not_change_default(monkeypatch):
    import i18n

    monkeypatch.setattr(i18n, "get_lang", lambda: "zh")
    assert mm.resolve_hub({"ui_lang": "zh"}) == "hf"


def test_function_defaults_are_hf():
    import inspect

    for fn in (mm.download_asr, mm.is_asr_cached, mm.get_local_model_path, mm.asr_model_id):
        assert inspect.signature(fn).parameters["hub"].default == "hf"
