import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Локальный прототип умного поиска по письмам.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--visitor-mail", action=argparse.BooleanOptionalAction, default=None,
                        help="Разрешить подключение почты на время сессии посетителя.")
    args = parser.parse_args()
    from .app import app, visitor_enabled

    app.state.visitor_enabled = visitor_enabled(args.port == 8766) if args.visitor_mail is None else args.visitor_mail
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
