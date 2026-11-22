"""Conservative one-edit correction using only word frequencies from the mail corpus."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
import re

from .query_variants import VARIANTS

_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
_WORD = re.compile(r"[а-яё]{6,32}")
_IN_TEXT = re.compile(r"(?<!\w)[А-ЯЁа-яё]{5,32}(?!\w)")
_TOKENS = re.compile(r'«[^»]*(?:»|$)|"[^"]*(?:"|$)|(?:https?://|www\.)\S+|\w+(?:[.@+/:-]\w+)*')
_CALENDAR = re.compile(r"январ|феврал|март|апрел|июн|июл|август|сентябр|октябр|ноябр|декабр|"
                       r"понедель|вторник|сред[ауые]|четвер|пятниц|суббот|воскресен|"
                       r"сегодня|вчера|позавчера|завтра|недел|месяц")


def _edits(word: str):
    for i in range(len(word) + 1):
        left, right = word[:i], word[i:]
        if right:
            yield left + right[1:]
        if len(right) > 1 and right[0] != right[1]:
            yield left + right[1] + right[0] + right[2:]
        for letter in _ALPHABET:
            yield left + letter + right
            if right and letter != right[0]:
                yield left + letter + right[1:]


class CorpusSpelling:
    """Preserve attested forms, known inflections, names, quoted terms and identifiers."""

    def __init__(self, texts: Iterable[str], stem: Callable[[str], str]):
        frequency, lowercase = Counter(), Counter()
        for text in texts:
            words = _IN_TEXT.findall(text.replace("ё", "е").replace("Ё", "Е"))
            frequency.update({word.lower() for word in words})
            lowercase.update({word for word in words if word.islower()})
        self.known = set(frequency)
        self.frequent = {w for w, count in frequency.items() if len(w) >= 6 and lowercase[w] >= 5 and 2 * lowercase[w] >= count}
        self.stem = stem
        self.roots = {stem(w) for w in self.known}

    def correct(self, query: str, transliterations=()) -> str:
        def replace(match: re.Match) -> str:
            original = match[0]
            if not _WORD.fullmatch(original):
                return original
            word = original.replace("ё", "е")
            if word in transliterations or word in VARIANTS or word in self.known or self.stem(word) in self.roots or _CALENDAR.match(word):
                return original
            result = None
            for neighbor in _edits(word):
                if neighbor in self.frequent:
                    if result is not None and neighbor != result:
                        return original
                    result = neighbor
            return result or original
        return _TOKENS.sub(replace, query)
