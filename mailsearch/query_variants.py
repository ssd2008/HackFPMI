"""Attested query variants for campus mail and technical correspondence."""
import re

# The sparse index shares features across common equivalent spellings.
# Kafka also names an author: only fall back to the product when no literal match exists.
_GROUPS = {
    "общежитие": "общага общаги общагу общаге общагой общагам общагах общаг",
    "лабораторная": "лаба лабу лабе лабы лабой лабах",
    "договор": "дог-р",
    "spark": "спарк спарка спарке спарком",
    "docker": "докер докера докере докером",
    "kafka": "кафка кафку кафки кафке кафкой",
    "postgresql": "постгрес постгреса постгресе постгресом postgres",
    "sql": "эскьюэль",
}
VARIANTS = {form: target for target, forms in _GROUPS.items() for form in forms.split()}
EQUIVALENTS = {form: target for form, target in VARIANTS.items() if target != "kafka"}
# Consume complete identifiers, addresses, URLs and literal phrases before testing words.
_CHUNKS = re.compile(r'«[^»]*(?:»|$)|"[^"]*(?:"|$)|(?:https?://|www\.)\S+|[\w%+]+(?:[.@+/:-][\w%+]+)*', re.I)


def expand_query(text, has_word):
    def replace(match):
        word = match[0].lower().replace("ё", "е")
        target = VARIANTS.get(word)
        if target and has_word(target) and (word in EQUIVALENTS or not has_word(word)):
            return target
        return match[0]

    return _CHUNKS.sub(replace, text)
