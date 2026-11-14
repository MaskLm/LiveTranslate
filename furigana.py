"""Hiragana furigana (readings above kanji) for Japanese recognized text.

Qt-free on purpose: this module holds the reading engine (fugashi + unidic-lite),
the kana/kanji alignment, the settings helpers and the pure line-breaking
function shared by the overlay and the subtitle window, so all of it is
unit-testable without a display.

    annotate("食べる") -> [Segment("食", "た"), Segment("べる", None)]
"""

import logging
import re
import threading
from functools import lru_cache
from typing import Callable, NamedTuple, Sequence

log = logging.getLogger(__name__)

MODES = ("ruby", "paren")

DEFAULTS = {
    "furigana_enabled": False,
    "furigana_mode": "ruby",       # "ruby" (reading above) | "paren" (漢字(かんじ))
    "furigana_scale": 0.5,         # ruby font size relative to the base font
    "furigana_overlay": True,      # apply to the overlay original text
    "furigana_subtitle": True,     # apply to the subtitle window original line
}
SCALE_MIN, SCALE_MAX = 0.3, 1.0

_JA_LANGS = {"ja", "jp", "jpn", "japanese"}


class Segment(NamedTuple):
    surface: str
    reading: str | None = None  # hiragana, only for segments that contain kanji


# ── Settings helpers ──

def normalize_settings(src: dict | None) -> dict:
    """Return the five furigana settings with defaults filled and values clamped."""
    src = src or {}
    out = dict(DEFAULTS)
    for key in ("furigana_enabled", "furigana_overlay", "furigana_subtitle"):
        if key in src:
            out[key] = bool(src[key])
    if src.get("furigana_mode") in MODES:
        out["furigana_mode"] = src["furigana_mode"]
    try:
        scale = float(src.get("furigana_scale", DEFAULTS["furigana_scale"]))
        if scale == scale:  # not NaN
            out["furigana_scale"] = round(min(SCALE_MAX, max(SCALE_MIN, scale)), 2)
    except (TypeError, ValueError):
        pass
    return out


def is_japanese(lang: str | None) -> bool:
    return (lang or "").strip().lower() in _JA_LANGS


def wanted(opts: dict, lang: str | None, target: str) -> bool:
    """True when furigana should be shown for `target` ("overlay"/"subtitle")."""
    return bool(
        opts.get("furigana_enabled")
        and opts.get(f"furigana_{target}", True)
        and is_japanese(lang)
    )


# ── Kana helpers ──

_KANJI_CLASS = "一-鿿㐀-䶿々〆ヶ豈-﫿\U00020000-\U0002a6df"
_KANJI_RE = re.compile(f"[{_KANJI_CLASS}]")
_RUN_RE = re.compile(f"([{_KANJI_CLASS}]+)")


def has_kanji(text: str) -> bool:
    return bool(_KANJI_RE.search(text))


def kata_to_hira(text: str) -> str:
    """Katakana -> hiragana (U+30A1..U+30F6 shifted by 0x60); everything else, incl. ー, kept."""
    return "".join(
        chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text
    )


def align_reading(surface: str, reading: str) -> list[Segment]:
    """Split `surface` into kanji runs (with the matching part of `reading`) and
    okurigana/kana runs (no reading).

    `reading` must be hiragana. Matching is a reluctant regex over the kana runs, so
    取り扱い/とりあつかい -> 取[と] り 扱[あつか] い. If the kana runs do not line up
    with the reading, the whole token falls back to one ruby group.
    """
    if not has_kanji(surface):
        return [Segment(surface)]
    if not reading:
        return [Segment(surface)]
    parts = [p for p in _RUN_RE.split(surface) if p]
    pattern = "".join(
        "(.+?)" if _KANJI_RE.match(p[0]) else re.escape(kata_to_hira(p)) for p in parts
    )
    m = re.fullmatch(pattern, reading)
    if m is None:
        return [Segment(surface, reading)]
    groups = iter(m.groups())
    out = []
    for p in parts:
        if _KANJI_RE.match(p[0]):
            out.append(Segment(p, next(groups)))
        else:
            out.append(Segment(p))
    return out


# ── Tagger (lazy singleton) ──

_lock = threading.Lock()
_tagger = None
_unavailable = False
_warned = False


def _warn_once(msg: str, exc: BaseException | None = None):
    global _warned
    if not _warned:
        _warned = True
        log.warning("Furigana disabled: %s%s", msg, f" ({exc!r})" if exc else "")


def _get_tagger():
    """Caller must hold _lock."""
    global _tagger, _unavailable
    if _tagger is None and not _unavailable:
        try:
            import fugashi  # noqa: PLC0415

            _tagger = fugashi.Tagger()
        except Exception as exc:  # noqa: BLE001
            _unavailable = True
            _warn_once("fugashi/unidic-lite unavailable", exc)
    return _tagger


def _merge_plain(segs: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        if s.reading is None and out and out[-1].reading is None:
            out[-1] = Segment(out[-1].surface + s.surface)
        else:
            out.append(s)
    return out


@lru_cache(maxsize=512)
def _annotate_cached(text: str) -> tuple[Segment, ...]:
    if not has_kanji(text):
        return (Segment(text),)
    try:
        with _lock:
            tagger = _get_tagger()
            if tagger is None:
                return (Segment(text),)
            tokens = [(w.surface, getattr(w.feature, "kana", None)) for w in tagger(text)]
    except Exception as exc:  # noqa: BLE001
        _warn_once("reading analysis failed", exc)
        return (Segment(text),)
    segs: list[Segment] = []
    pos = 0
    for surface, kana in tokens:
        # MeCab drops nothing, but keep the original text intact if it ever does.
        idx = text.find(surface, pos)
        if idx > pos:
            segs.append(Segment(text[pos:idx]))
        if idx < 0:
            continue
        pos = idx + len(surface)
        if kana and kana != "*" and has_kanji(surface):
            segs.extend(align_reading(surface, kata_to_hira(kana)))
        else:
            segs.append(Segment(surface))
    if pos < len(text):
        segs.append(Segment(text[pos:]))
    return tuple(_merge_plain(segs))


def annotate(text: str) -> list[Segment]:
    """Annotate `text`; never raises, and always concatenates back to `text`."""
    if not text:
        return []
    try:
        return list(_annotate_cached(text))
    except Exception as exc:  # noqa: BLE001
        _warn_once("annotate failed", exc)
        return [Segment(text)]


def has_ruby(segments: Sequence[Segment]) -> bool:
    return any(s.reading for s in segments)


def to_paren_text(segments: Sequence[Segment], open_: str = "(", close: str = ")") -> str:
    """漢字(かんじ) inline form."""
    return "".join(
        f"{s.surface}{open_}{s.reading}{close}" if s.reading else s.surface
        for s in segments
    )


# ── Line breaking (shared by overlay and subtitle window) ──

BREAK_AFTER = " ,，。、!！?？;；:：.　"
_WORD_RE = re.compile(r"[A-Za-z0-9_'’\-]+|.", re.DOTALL)


def split_plain(text: str) -> list[str]:
    """Break plain text into wrap units: ASCII words stay whole, other chars are single."""
    return _WORD_RE.findall(text)


def wrap_units(segments: Sequence[Segment]) -> list[Segment]:
    """Flatten segments into atomic wrap units: ruby groups stay whole, plain text is split."""
    units: list[Segment] = []
    for s in segments:
        if s.reading:
            units.append(s)
        else:
            units.extend(Segment(p) for p in split_plain(s.surface))
    return units


def breaks_after(text: str) -> bool:
    return bool(text) and (text[-1] in BREAK_AFTER)


def wrap_lines(
    widths: Sequence[float], breaks: Sequence[bool], avail: float
) -> list[tuple[int, int]]:
    """Greedy line breaking over atomic units; returns [(start, end), ...] index ranges.

    A line is broken at the last break opportunity in its second half if one exists,
    otherwise right before the overflowing unit. Blank units are trimmed at line
    edges. A unit wider than `avail` still gets a line of its own.
    """
    n = len(widths)
    lines: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j, w = i, 0.0
        while j < n and (j == i or w + widths[j] <= avail):
            w += widths[j]
            j += 1
        if j >= n:
            end = n
            nxt = n
        else:
            end = j
            for k in range(j - 1, i + (j - i) // 2 - 1, -1):
                if k >= i and breaks[k]:
                    end = k + 1
                    break
            nxt = end
        lines.append((i, end))
        i = nxt
    return lines


def wrap_segments(
    segments: Sequence[Segment], measure: Callable[[Segment], float], avail: float
) -> list[list[Segment]]:
    """Wrap annotated text into lines of segments; ruby groups are never split.

    `measure(seg)` returns the advance width of a unit (for a ruby group: the max of
    the base and the reading widths). Blank units at line edges are dropped.
    """
    units = wrap_units(segments)
    if not units:
        return []
    widths = [measure(u) for u in units]
    breaks = [breaks_after(u.surface) for u in units]
    out: list[list[Segment]] = []
    for a, b in wrap_lines(widths, breaks, avail):
        line = units[a:b]
        while line and not line[0].reading and not line[0].surface.strip():
            line = line[1:]
        while line and not line[-1].reading and not line[-1].surface.strip():
            line = line[:-1]
        if line:
            out.append(line)
    return out
