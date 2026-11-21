"""Recognize an explicit count of a named unit in Russian or mixed-script mail."""
from functools import lru_cache
import re
import threading

_UNIT_QUESTION = re.compile(r"\bскольк(?:о|их)\s+((?:[а-яёa-z-]+\s*){1,2})", re.I)
_NUMBER_UNIT = re.compile(r"(?<![\w.,:/-])(\d+(?:[.,]\d+)?)(?!\d|[.,:/-]\d)[ \t]+"
                          r"([а-яёa-z][а-яёa-z-]+(?:[ \t]+[а-яёa-z][а-яёa-z-]+){0,2})", re.I)
_MODIFIER = re.compile(r"free|additional|extra|unused|new|bonus|trial|paid|[a-z]+ed|[A-Z]{2,5}|"
                       r"[а-яё]+(?:ых|их|ые|ие|ая|ого|ую|ый|ой|ий)")
_TRANSLITERATE = str.maketrans(dict(zip("абвгдежзийклмнопрстуфхцчшщъыьэюя", (
    "a", "b", "v", "g", "d", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s",
    "t", "u", "f", "h", "ts", "ch", "sh", "sh", "", "y", "", "e", "yu", "ya"), strict=True)))
_thread = threading.local()


@lru_cache(maxsize=8000)
def stem(word: str) -> str:
    if not hasattr(_thread, "stemmer"):
        import snowballstemmer
        _thread.stemmer = snowballstemmer.stemmer("russian")
    return _thread.stemmer.stemWord(word)


@lru_cache(maxsize=1024)
def _units(query: str) -> frozenset[str]:
    question = _UNIT_QUESTION.search(query.casefold().replace("ё", "е"))
    if not question:
        return frozenset()
    words = [w for w in question[1].split() if len(w) > 2 and w not in {"мне", "было", "был", "всего"}]
    return frozenset(stem(w) for w in words[:2])


def quantities_requested(query: str) -> bool:
    return bool(_units(query))


def counted_values(query: str, text: str, *, across_scripts: bool = False):
    """Yield (start, numeric text) for counts adjacent to a noun in the question."""
    units = _units(query)
    if not units:
        return
    cognates = {w.translate(_TRANSLITERATE) for w in units if len(w) >= 4 and not w.isascii()}
    for match in _NUMBER_UNIT.finditer(text):
        for raw in match[2].split():
            word = raw.casefold().replace("ё", "е")
            latin = re.sub(r"c(?=[aourl])", "k", word.removesuffix("s").replace("ph", "f"))
            if ((not across_scripts and stem(word) in units) or (word.isascii() and latin in cognates)):
                yield match.start(), match[1]
                break
            if not (_MODIFIER.fullmatch(raw) or _MODIFIER.fullmatch(word)):
                break


def has_counted_quantity(query: str, text: str, *, across_scripts: bool = False) -> bool:
    """Test for the requested number/unit. Purchase, gift and other relations are separate."""
    return next(counted_values(query, text, across_scripts=across_scripts), None) is not None
