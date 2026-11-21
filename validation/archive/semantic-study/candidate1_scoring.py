"""Text sent to an embedding model for open-ended searches."""
import re


_REQUEST = re.compile(
    r"^\s*(?:(?:please\s+)?(?:find|look\s+for|search\s+for|show(?:\s+me)?|retrieve|list|locate)"
    r"|найди(?:те)?|покажи(?:те)?|найти|отыщи(?:те)?|ищу)\b[\s,:-]*", re.I
)
_MAIL_TOPIC = re.compile(
    r"^\s*(?:(?:my|the|all|any)\s+)*(?:(?:e-?mails?|messages?|letters?)\s+"
    r"(?:about|regarding|concerning|related\s+to)|(?:письм[а-яё]*|сообщени[а-яё]+|уведомлени[а-яё]+)"
    r"\s+(?:о|об|про))\s+", re.I
)
_QUESTION = re.compile(
    r"^(?:(?:в|на|из|с|по|под|от|для|у)\s+)?(?:как(?:ой|ая|ое|ие|ую|ом|им|их|ого|ому|ими|ов[аоы]?)?|что|чего|чем|чему|чь[а-яё]+|"
    r"кто|кого|кому|кем|когда|зачем|почему|скольк[а-яё]*|где|куда|котор[а-яё]*|"
    r"what|where|when|who(?:se|m)?|which|how|why)\b", re.I
)


def semantic_intent(query: str) -> tuple[str, float]:
    """Keep names, punctuation and word order. Questions receive more weight on meaning."""
    original = query.strip()
    prefix = _REQUEST.match(original)
    topic = _MAIL_TOPIC.sub("", original[prefix.end():]).strip() if prefix else original
    topic = topic or original
    is_question = topic.endswith("?") or bool(_QUESTION.match(topic))
    return topic, 0.08 if is_question else 0.20
