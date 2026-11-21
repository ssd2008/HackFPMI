"""Prepare the pinned Sergey mailbox for the local app. Run with the project Python."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory, TemporaryFile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.import_mail_export import write_corpus

REVISION = "1c6a47129299f0887d5786d314010d06d6be4729"
SOURCE = "data/Sergey-mails.jsonl"
BLOB = "8f29fdb126f4af6c03b531c5f0556ce0c7680278"
SOURCE_SHA256 = "c6f49b3d64d4fef54cd75cbe7b98cdd508a40177377404aa9fa0e4edb05fb9d8"
CORPUS_SHA256 = "7b810045963ae633ae26846f5b15331842f9d0892c2cd4bd7da5e837f481d38f"
NAMESPACE = "independent-export-01"
COUNT = 5939


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    permissions = path.lstat()
    if not stat.S_ISDIR(permissions.st_mode) or permissions.st_uid != os.getuid():
        raise ValueError(f"Expected a private directory owned by this account: {path}")
    if permissions.st_mode & 0o077:
        raise ValueError(f"Private mail directory requires mode 0700: {path}")
    return path


def sha256(path):
    with path.open("rb") as contents:
        return hashlib.file_digest(contents, "sha256").hexdigest()


def prepare_corpus(local):
    box = private_directory(local / "sergey")
    destination = box / "documents.jsonl"
    git = ["git", "--no-replace-objects", "-C", str(ROOT)]
    source = f"{REVISION}:{SOURCE}"
    if subprocess.check_output([*git, "rev-parse", "--verify", source], text=True).strip() != BLOB:
        raise ValueError("The pinned commit does not contain the expected mail blob.")
    with TemporaryDirectory(dir=box, prefix=".prepare-") as temporary:
        # Read the object itself. The working tree may have a different export.
        with TemporaryFile(dir=temporary) as raw:
            subprocess.run([*git, "cat-file", "blob", BLOB], stdout=raw, check=True)
            raw.seek(0)
            if hashlib.file_digest(raw, "sha256").hexdigest() != SOURCE_SHA256:
                raise ValueError("The source export checksum differs from the pinned snapshot.")
            raw.seek(0)
            stats = write_corpus(raw, Path(temporary), NAMESPACE, ZoneInfo("Europe/Moscow"))
        staged = Path(temporary) / "documents.jsonl"
        if stats["documents"] != COUNT or stats["mailboxes"] != 1 or sha256(staged) != CORPUS_SHA256:
            raise ValueError("The normalized export differs from the original per-message IDs or content.")
        # Publish an entirely verified snapshot. Idempotent and safe if two
        # local processes prepare the same mailbox at the same time.
        try:
            os.link(staged, destination)
        except FileExistsError:
            if not stat.S_ISREG(destination.lstat().st_mode) or sha256(destination) != CORPUS_SHA256:
                raise ValueError("A different file already occupies the private mailbox destination.") from None
        destination.chmod(0o600)
    return destination, stats


def prepare_index(local, source):
    import numpy as np
    from filelock import FileLock
    from mailsearch.dense import DIMENSION, _index_name
    from mailsearch.retrieval import SearchEngine

    with source.open(encoding="utf-8") as data:
        documents = [json.loads(line) for line in data if line.strip()]
    indexes = private_directory(local / "indexes")
    name = _index_name(documents)
    target = indexes / name
    previous = ROOT / ".cache" / "mailsearch" / name
    with FileLock(str(target) + ".lock", mode=0o600):
        if not target.exists() and previous.is_file():
            # The name fingerprints all IDs, contents, ordering and model
            # revision. Check the whole array before reusing the local cache.
            vectors = np.load(previous, mmap_mode="r", allow_pickle=False)
            if vectors.shape != (len(documents), DIMENSION) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                raise ValueError("The cached vectors failed validation.")
            with NamedTemporaryFile(dir=indexes, prefix=".copy-") as staged:
                with previous.open("rb") as data:
                    shutil.copyfileobj(data, staged)
                staged.flush()
                os.link(staged.name, target)
    engine = SearchEngine(documents, indexes)
    if set(engine.boxes) != {documents[0]["synthetic_user"]} or len(engine.by_id) != COUNT:
        raise ValueError("The prepared search engine does not match the pinned mailbox.")
    target.chmod(0o600)
    return list(engine.dense.vectors.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=ROOT / ".cache/local-mail", help="Private local mail directory.")
    parser.add_argument("--skip-index", action="store_true", help="Prepare documents without loading the search models.")
    args = parser.parse_args()
    os.umask(0o077)
    local = private_directory(args.local_dir.expanduser().absolute())
    destination, stats = prepare_corpus(local)
    # Read the ID from the verified corpus so the app keeps the old namespace.
    with destination.open(encoding="utf-8") as data:
        mailbox = json.loads(data.readline())["synthetic_user"]
    info = {**stats, "mailbox": mailbox, "corpus_sha256": CORPUS_SHA256}
    if not args.skip_index:
        info["index_shape"] = prepare_index(local, destination)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
