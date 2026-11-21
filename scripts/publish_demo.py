"""Publish the committed static frontend from HackFPMI main to GitHub Pages."""

import argparse
from html import escape, unescape
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEMO_REPO = "https://github.com/ssd2008/HackFPMI-demo.git"
META_ORIGIN = re.compile(r'<meta name="mailsearch-api-origin" content="([^"]*)"[^>]*>')
OBJECT_ID = re.compile(r"[0-9a-f]{40}")
ASSET_SUFFIXES = {".avif", ".css", ".gif", ".ico", ".jpeg", ".jpg", ".js", ".mjs",
                  ".png", ".svg", ".webp", ".woff", ".woff2"}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def render_frontend(revision, demo, api_origin):
    if api_origin is None:
        deployed_html = (demo / "index.html").read_text(encoding="utf-8")
        previous_config = META_ORIGIN.search(deployed_html)
        if previous_config is None:
            raise SystemExit("В HackFPMI-demo не найден тег с адресом API.")
        api_origin = unescape(previous_config[1])

    frontend = {}
    entries = git(ROOT, "ls-tree", "-r", "-z", revision, "--", "static/").split(b"\0")
    for entry in filter(None, entries):
        metadata, raw_path = entry.split(b"\t", 1)
        path = Path(raw_path.decode())
        mode, _, object_id = metadata.decode().split()
        if mode not in {"100644", "100755"} or any(part.startswith(".") for part in path.parts):
            raise SystemExit(f"В статике найден скрытый файл или ссылка: {path}")
        if path != Path("static/index.html") and path.suffix not in ASSET_SUFFIXES:
            raise SystemExit(f"Проверьте, что этот файл предназначен для публикации: {path}")
        frontend[path] = git(ROOT, "cat-file", "blob", object_id)

    index = frontend.pop(Path("static/index.html")).decode()
    tag = f'<meta name="mailsearch-api-origin" content="{escape(api_origin, quote=True)}" data-external>'
    index, count = META_ORIGIN.subn(lambda _: tag, index)
    if count != 1:
        raise SystemExit("В static/index.html должен быть один тег mailsearch-api-origin.")

    # Pages caches HTML and assets independently. Keep each release at a unique
    # path, plus the previous release for browsers still holding its HTML.
    assets = demo / "static"
    assets.mkdir(exist_ok=True)
    version_file = demo / "version.txt"
    previous = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else ""
    if previous != revision:
        for directory in assets.iterdir():
            if directory.is_dir() and OBJECT_ID.fullmatch(directory.name) and directory.name not in {previous, revision}:
                shutil.rmtree(directory)

    index = index.replace('"./static/', f'"./static/{revision}/')
    for path, content in frontend.items():
        target = assets / revision / path.relative_to("static")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (demo / "index.html").write_text(index, encoding="utf-8")
    (demo / ".nojekyll").touch()
    version_file.write_text(revision + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-origin", help="Задать новый HTTPS-адрес API. По умолчанию сохраняется текущий.")
    args = parser.parse_args()
    api_origin = args.api_origin
    if api_origin is not None:
        try:
            url = urlsplit(api_origin)
            valid_port = url.port != 0
        except ValueError:
            parser.error("--api-origin должен быть HTTPS-адресом без пути, например https://api.example.org.")
        if (url.scheme != "https" or not url.hostname or url.username is not None or not valid_port
                or url.path not in ("", "/") or "?" in api_origin or "#" in api_origin
                or any(char.isspace() for char in api_origin)):
            parser.error("--api-origin должен быть HTTPS-адресом без пути, например https://api.example.org.")
        api_origin = api_origin.rstrip("/")

    git(ROOT, "fetch", "origin", "main")
    revision = git(ROOT, "rev-parse", "FETCH_HEAD").decode().strip()
    if revision != git(ROOT, "rev-parse", "HEAD").decode().strip():
        raise SystemExit("Обновите локальную копию до актуального origin/main.")

    with tempfile.TemporaryDirectory(prefix="hackfpmi-pages-") as workdir:
        demo = Path(workdir) / "demo"
        git(ROOT, "clone", "--depth=1", "--branch=main", DEMO_REPO, str(demo))
        render_frontend(revision, demo, api_origin)
        git(demo, "add", "-A", "--", "index.html", "static", ".nojekyll", "version.txt")
        if not git(demo, "status", "--porcelain"):
            print(f"GitHub Pages уже содержит {revision}.")
            return
        print(git(demo, "commit", "-m", f"Publish frontend from {revision[:10]}").decode().strip())
        git(demo, "push", "origin", "main")
        print(f"Версия {revision} отправлена в GitHub Pages.")


if __name__ == "__main__":
    main()
