#!/bin/sh
set -eu
cd "$(dirname "$0")"
if ! test -x .venv/bin/python; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 .venv
    else
        "${PYTHON:-python3.12}" -m venv .venv
    fi
fi
if ! cmp -s requirements.txt .venv/.mailsearch-requirements; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python .venv/bin/python -r requirements.txt
    else
        .venv/bin/python -m pip install -r requirements.txt
    fi
    cp requirements.txt .venv/.mailsearch-requirements
fi
if [ "${MAILSEARCH_GENERATIVE_ANSWERS:-0}" = "1" ]; then
    .venv/bin/python -m mailsearch.prepare --models-only
else
    .venv/bin/python -m mailsearch.prepare --models-only --search-only
fi
exec .venv/bin/python -m mailsearch "$@"
