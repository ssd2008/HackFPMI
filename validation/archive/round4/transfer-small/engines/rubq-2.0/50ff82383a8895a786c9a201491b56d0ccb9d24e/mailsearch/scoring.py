"""Recognize natural-language information requests for hybrid ranking."""
import re


_SEARCH = re.compile(
    r"^(?:(?:please\s+)?(?:find|look\s+for|search\s+for|show|retrieve|list|locate)|"
    r"(?:пожалуйста[, ]+)?(?:найди(?:те)?|найти|покажи(?:те)?|отыщи(?:те)?|ищу|перечисли(?:те)?|"
    r"помоги(?:те)?\s+найти|хочу\s+найти|мне\s+нуж[а-яё]*))\b", re.I
)
_QUESTION = re.compile(
    r"^(?:(?:в|на|из|с|по|под|от|для|у)\s+)?(?:как(?:ой|ая|ое|ие|ую|ом|им|их|ого|ому|ими|ов[аоы]?)?|"
    r"что|чего|чем|чему|чь[а-яё]+|кто|кого|кому|кем|когда|зачем|почему|скольк[а-яё]*|где|куда|"
    r"котор[а-яё]*|what|where|when|who(?:se|m)?|which|how|why)\b", re.I
)


def semantic_lexical_scale(query: str) -> float | None:
    """Keep command and keyword searches on the proximity ranker."""
    text = query.strip()
    if _SEARCH.match(text):
        return None
    if text.endswith("?") or _QUESTION.match(text):
        return 0.08
    if len(text.split()) >= 6:
        return 0.20
    return None
