import argparse
from pathlib import Path

import uvicorn

from .store import DATA


def main():
    parser = argparse.ArgumentParser(description="Локальный прототип умного поиска по письмам.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path, default=DATA, help="Каталог с documents.jsonl.")
    args = parser.parse_args()
    from .app import app

    app.state.data_dir = args.data.resolve()
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
