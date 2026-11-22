"""Dictionary aliases and corpus-backed transliteration for mail search."""
from collections import defaultdict
from pathlib import Path
import re


def _read_dictionary():
    aliases, fallback, phrases = {}, {}, {}
    groups = {"alias": aliases, "fallback": fallback, "phrase": phrases}
    path = Path(__file__).with_name("resources") / "query_variants.tsv"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            kind, target, forms = line.lower().replace("ё", "е").split("\t")
            for form in forms.split():
                groups[kind][form] = target
    return aliases, fallback, phrases


EQUIVALENTS, _FALLBACK, PHRASES = _read_dictionary()
VARIANTS = {**EQUIVALENTS, **_FALLBACK, **PHRASES}
# Keep punctuated abbreviations whole, without splitting IDs, email addresses or URLs.
ABBREVIATION_PATTERN = "|".join(re.escape(w) for w in VARIANTS if not w.isalpha())
_CHUNKS = re.compile(r'«[^»]*(?:»|$)|"[^"]*(?:"|$)|(?:https?://|www\.)\S+|[\w%+]+(?:[.@+/:-][\w%+]+)*', re.I)

# Latin-to-Cyrillic letter groups, longest first. No corpus or language model is downloaded.
_LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz",
                    "а б к д е ф г х и дж к л м н о п к р с т у в в кс и з".split(), strict=True))
_DIGRAPHS = {
    "shch": "щ", "sch": "щ", "tch": "ч", "ouse": "аус", "zh": "ж", "kh": "х",
    "sh": "ш", "ch": "ч", "ph": "ф", "th": "т", "ck": "к", "ts": "ц",
    "ya": "я", "yu": "ю", "yo": "е", "ye": "е", "ee": "и", "oo": "у", "qu": "кв",
}
_PART = re.compile("|".join(sorted(_DIGRAPHS, key=len, reverse=True)) + "|[a-z]")
_LATIN = re.compile("[a-z]{4,32}")
_CYRILLIC = re.compile("[а-я]{4,48}")


def to_cyrillic(word):
    def letter(match):
        part = match[0]
        if part == "c" and word[match.end():match.end()+1] in ("e", "i", "y"):
            return "с"
        return _DIGRAPHS.get(part) or _LETTERS[part]

    return _PART.sub(letter, word)


def transliteration_index(vocabulary):
    names = defaultdict(list)
    for term in vocabulary:
        if _LATIN.fullmatch(term):
            names[to_cyrillic(term)].append(term)
    return dict(names)


def expand_query(text, has_word, names=None):
    def replace(match):
        word = match[0].lower().replace("ё", "е")
        if word in PHRASES:
            return PHRASES[word] if has_word(word) else match[0]
        target = VARIANTS.get(word)
        if target:
            if has_word(target) and (word in EQUIVALENTS or not has_word(word)):
                return target
            return match[0]
        if names is None:
            return match[0]
        if _CYRILLIC.fullmatch(word):
            candidates = names.get(word, ())
        elif _LATIN.fullmatch(word):
            candidates = [to_cyrillic(word)]
        else:
            return match[0]
        if not candidates or has_word(word):
            return match[0]
        visible = [candidate for candidate in candidates if has_word(candidate)]
        return visible[0] if len(visible) == 1 else match[0]

    return _CHUNKS.sub(replace, text)
