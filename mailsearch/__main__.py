import argparse
from os import getenv
from pathlib import Path

import uvicorn

from .store import DATA


def main():
    parser = argparse.ArgumentParser(description="Локальный прототип умного поиска по письмам.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path, help="Каталог с documents.jsonl.")
    parser.add_argument("--local-mail", action=argparse.BooleanOptionalAction, default=None,
                        help="Личные ящики, синхронизация и отправка. По умолчанию включены на 8766 с основным корпусом.")
    parser.add_argument("--local-mail-dir", type=Path, help="Каталог личных ящиков на этом компьютере.")
    args = parser.parse_args()
    from .app import app

    from .local_mail import LOCAL_DATA
    local = args.local_mail
    if local is None:
        local = args.port == 8766 and args.data is None and not getenv("MAILSEARCH_FRONTEND_ORIGIN", "").strip()
    if local and getenv("MAILSEARCH_FRONTEND_ORIGIN", "").strip():
        parser.error("Локальные ящики нельзя включать в публичном демо.")
    app.state.data_dir = (args.data or DATA).resolve()
    app.state.local_dir = (args.local_mail_dir or LOCAL_DATA).resolve() if local else None
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, proxy_headers=not local)


if __name__ == "__main__":
    main()
