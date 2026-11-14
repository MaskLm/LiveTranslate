import json
import time

import pytest

import furigana
from furigana import Segment, align_reading, annotate

pytest.importorskip("fugashi")
pytest.importorskip("unidic_lite")


def _join(segs):
    return "".join(s.surface for s in segs)


@pytest.mark.parametrize(
    "surface,reading,expected",
    [
        ("食べる", "たべる", [Segment("食", "た"), Segment("べる")]),
        ("取り扱い", "とりあつかい",
         [Segment("取", "と"), Segment("り"), Segment("扱", "あつか"), Segment("い")]),
        ("振り返り", "ふりかえり",
         [Segment("振", "ふ"), Segment("り"), Segment("返", "かえ"), Segment("り")]),
        ("お願い", "おねがい", [Segment("お"), Segment("願", "ねが"), Segment("い")]),
        ("東京", "とうきょう", [Segment("東京", "とうきょう")]),
        ("人々", "ひとびと", [Segment("人々", "ひとびと")]),
        # katakana okurigana in the surface matches a hiragana reading
        ("見ル", "みる", [Segment("見", "み"), Segment("ル")]),
        # kana runs do not line up -> whole token as one ruby group
        ("食べる", "のむ", [Segment("食べる", "のむ")]),
        ("かな", "かな", [Segment("かな")]),
        ("漢字", "", [Segment("漢字")]),
    ],
)
def test_align_reading(surface, reading, expected):
    assert align_reading(surface, reading) == expected


def test_katakana_to_hiragana():
    assert furigana.kata_to_hira("トウキョウ") == "とうきょう"
    assert furigana.kata_to_hira("ヴァイオリン") == "ゔぁいおりん"
    assert furigana.kata_to_hira("コーヒー") == "こーひー"  # long mark kept
    assert furigana.kata_to_hira("abc漢") == "abc漢"


def test_mixed_sentence():
    text = "今日は東京で日本語を勉強します"
    segs = annotate(text)
    assert _join(segs) == text
    readings = {s.surface: s.reading for s in segs if s.reading}
    assert readings["今日"] == "きょう"  # not きょー: kana form, not pronunciation
    assert readings["東京"] == "とうきょう"
    assert readings["日本"] == "にっぽん"
    assert readings["勉強"] == "べんきょう"
    # particles / kana stay unannotated
    assert all(s.reading is None for s in segs if s.surface in ("は", "で", "を"))


def test_okurigana_split_in_sentence():
    segs = annotate("食べられなかった")
    assert segs[0] == Segment("食", "た")
    assert _join(segs) == "食べられなかった"
    assert annotate("取り扱いに注意してください")[:3] == [
        Segment("取", "と"), Segment("り"), Segment("扱", "あつか")]


@pytest.mark.parametrize("text", ["ひらがなだけ", "カタカナ", "Hello world", "123 !?。、", ""])
def test_no_kanji_passthrough(text):
    segs = annotate(text)
    assert not furigana.has_ruby(segs)
    assert _join(segs) == text


def test_latin_and_digits_unannotated_inside_kanji_text():
    segs = annotate("2024年にiPhoneを買った")
    assert _join(segs) == "2024年にiPhoneを買った"
    assert all(s.reading is None for s in segs if s.surface.isascii())


def test_error_fallback_never_raises(monkeypatch):
    furigana._annotate_cached.cache_clear()

    class Boom:
        def __call__(self, text):
            raise RuntimeError("mecab exploded")

    monkeypatch.setattr(furigana, "_tagger", Boom())
    monkeypatch.setattr(furigana, "_warned", False)
    segs = annotate("漢字の読み")
    assert segs == [Segment("漢字の読み")]
    furigana._annotate_cached.cache_clear()


def test_unavailable_tagger_fallback(monkeypatch):
    furigana._annotate_cached.cache_clear()
    monkeypatch.setattr(furigana, "_tagger", None)
    monkeypatch.setattr(furigana, "_unavailable", True)
    assert annotate("漢字") == [Segment("漢字")]
    furigana._annotate_cached.cache_clear()


def test_unknown_word_has_no_ruby():
    # token without a reading (e.g. digits/foreign) must not get ruby
    segs = annotate("𠮷野家へ行く")
    assert _join(segs) == "𠮷野家へ行く"


def test_performance_under_5ms():
    texts = [f"今日は東京で日本語を勉強しますよ、それから大阪にも行きます{'。' * i}" for i in range(40)]
    annotate(texts[0])  # load dictionary once (lazy)
    t0 = time.perf_counter()
    for t in texts:
        annotate(t)
    per_call_ms = (time.perf_counter() - t0) / len(texts) * 1000
    assert per_call_ms < 5, per_call_ms
    # cached hit is effectively free
    t0 = time.perf_counter()
    for _ in range(100):
        annotate(texts[0])
    assert (time.perf_counter() - t0) / 100 * 1000 < 0.5


def test_annotate_returns_independent_list():
    a = annotate("漢字")
    a.clear()
    assert annotate("漢字") != []


# ── settings ──

def test_settings_defaults_and_roundtrip():
    d = furigana.normalize_settings({})
    assert d == furigana.DEFAULTS
    assert d["furigana_enabled"] is False and d["furigana_mode"] == "ruby"
    assert d["furigana_scale"] == 0.5
    stored = json.loads(json.dumps({"other": 1, **d}))
    assert furigana.normalize_settings(stored) == d
    changed = {**d, "furigana_enabled": True, "furigana_mode": "paren",
               "furigana_scale": 0.65, "furigana_subtitle": False}
    assert furigana.normalize_settings(json.loads(json.dumps(changed))) == changed


def test_settings_clamp_and_invalid():
    n = furigana.normalize_settings
    assert n({"furigana_scale": 9})["furigana_scale"] == furigana.SCALE_MAX
    assert n({"furigana_scale": 0})["furigana_scale"] == furigana.SCALE_MIN
    assert n({"furigana_scale": "x"})["furigana_scale"] == 0.5
    assert n({"furigana_scale": float("nan")})["furigana_scale"] == 0.5
    assert n({"furigana_mode": "bogus"})["furigana_mode"] == "ruby"
    assert n(None) == furigana.DEFAULTS


def test_wanted_requires_japanese_and_flags():
    on = furigana.normalize_settings({"furigana_enabled": True})
    assert furigana.wanted(on, "ja", "overlay")
    assert furigana.wanted(on, "JA", "subtitle")
    assert not furigana.wanted(on, "en", "overlay")
    assert not furigana.wanted(on, None, "overlay")
    assert not furigana.wanted(furigana.normalize_settings({}), "ja", "overlay")
    off_sub = furigana.normalize_settings({"furigana_enabled": True, "furigana_subtitle": False})
    assert not furigana.wanted(off_sub, "ja", "subtitle")
    assert furigana.wanted(off_sub, "ja", "overlay")


def test_paren_text():
    segs = annotate("食べる本")
    assert furigana.to_paren_text(segs) == "食(た)べる本(ほん)"


# ── line breaking keeps ruby groups intact ──

def _measure(seg):
    # CJK char = 10px; ruby reading = 5px/char (half scale); group = max of both
    base = 10 * len(seg.surface)
    return max(base, 5 * len(seg.reading)) if seg.reading else base


def test_wrap_never_splits_ruby_group():
    segs = annotate("今日は東京で日本語を勉強します")
    for avail in (30, 45, 60, 80, 120, 500):
        lines = furigana.wrap_segments(segs, _measure, avail)
        flat = [u for ln in lines for u in ln]
        assert "".join(u.surface for u in flat) == "今日は東京で日本語を勉強します"
        # every ruby group of the input survives whole, with its reading
        in_groups = [s for s in segs if s.reading]
        out_groups = [u for u in flat if u.reading]
        assert out_groups == in_groups
        for ln in lines:
            w = sum(_measure(u) for u in ln)
            assert w <= avail or len(ln) == 1, (avail, ln)


def test_wrap_single_line_when_it_fits():
    segs = annotate("今日は東京")
    assert len(furigana.wrap_segments(segs, _measure, 1000)) == 1


def test_wrap_prefers_punctuation_and_keeps_ascii_words():
    segs = [Segment("今日は、"), Segment("東京", "とうきょう"), Segment("で hello world")]
    lines = furigana.wrap_segments(segs, _measure, 60)
    texts = ["".join(u.surface for u in ln) for ln in lines]
    assert "".join(texts).replace(" ", "") == "今日は、東京でhelloworld"
    assert texts[0].endswith("、")
    assert all(not t.startswith(" ") and not t.endswith(" ") for t in texts)
    # an ASCII word is never cut in the middle
    assert all(w in {"hello", "world"} or not w.isascii() for t in texts for w in t.split() if w.isascii())


def test_wrap_oversized_unit_gets_own_line():
    segs = [Segment("あ"), Segment("特別", "とくべつ"), Segment("い")]
    lines = furigana.wrap_segments(segs, _measure, 15)
    assert any(len(ln) == 1 and ln[0].reading for ln in lines)


def test_wrap_empty():
    assert furigana.wrap_segments([], _measure, 100) == []
