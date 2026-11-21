#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

if [ "$(git branch --show-current)" != main ] || [ -n "$(git status --porcelain)" ]; then
    echo "Запустите из отдельной чистой копии HackFPMI на ветке main." >&2
    exit 1
fi

git pull --ff-only origin main
revision=$(git rev-parse HEAD)
session=hackfpmi-backend
stamp=.cache/preview-revision
log=.cache/preview-backend.log
mkdir -p .cache
running=$(cat "$stamp" 2>/dev/null || :)

if ! tmux has-session -t "=$session" 2>/dev/null ||
   [ -z "$running" ] ||
   ! git diff --quiet "$running" "$revision" -- mailsearch run.sh requirements.txt data/documents.jsonl; then
    if tmux has-session -t "=$session" 2>/dev/null; then
        tmux kill-session -t "=$session"
    fi
    for _ in 1 2 3 4 5; do
        if ! lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Порт 8765 занят. Остановите прежний сервер перед деплоем." >&2
        exit 1
    fi
    : > "$log"
    tmux new-session -d -s "$session" -c "$PWD" \
        'exec env MAILSEARCH_FRONTEND_ORIGIN=https://ssd2008.github.io caffeinate -i ./run.sh >> .cache/preview-backend.log 2>&1'
    printf '%s\n' "$revision" > "$stamp"
fi

echo "Ожидаем API. Журнал: $PWD/$log"
deadline=$(($(date +%s) + 1200))
until curl -fsS --max-time 2 http://127.0.0.1:8765/api/mailboxes -o /dev/null 2>/dev/null; do
    if ! tmux has-session -t "=$session" 2>/dev/null || [ "$(date +%s)" -ge "$deadline" ]; then
        tail -n 25 "$log" >&2
        exit 1
    fi
    sleep 2
done
if ! tmux has-session -t "=$session" 2>/dev/null; then
    tail -n 25 "$log" >&2
    exit 1
fi
.venv/bin/python scripts/publish_demo.py "$@"
