"""Keep memoized words and queries from authenticated requests out of shared caches."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from threading import local

_workers = local()
_private = ContextVar("mailsearch_request_caches", default=None)


@contextmanager
def private_request():
    caches = {}
    token = _private.set(caches)
    try:
        yield
    finally:
        caches.clear()
        _private.reset(token)


def text_cache(maxsize):
    def decorate(function):
        shared = lru_cache(maxsize)(function)

        @wraps(function)
        def call(*args):
            caches = _private.get()
            if caches is None:
                return shared(*args)
            if function not in caches:
                caches[function] = lru_cache(maxsize)(function)
            return caches[function](*args)

        return call
    return decorate


def stem_word(language, word):
    # Snowball also retains its last word in the mutable stemmer. Private requests own that object.
    caches = _private.get()
    if caches is None:
        caches = _workers.__dict__
    if language not in caches:
        import snowballstemmer
        caches[language] = snowballstemmer.stemmer(language)
    return caches[language].stemWord(word)
