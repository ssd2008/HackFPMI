"""Rebuild the bundled bilingual keyword dictionary from the unmodified FreeDict source archives.

Use --archives for a directory containing rus-eng.tar.xz and eng-rus.tar.xz.
No network connection is used. See mailsearch/resources/FREEDICT.md.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
from pathlib import Path
import re
import tarfile
import unicodedata
from xml.etree.ElementTree import iterparse

import snowballstemmer


ROOT = Path(__file__).resolve().parents[1]
SHA512 = {
    "rus-eng": "9adf7a5eda27ccfa3e496b86b2fd575fc0e4af7ac5626a219d44e4ebfd56280deaf67cefd995ab04e9d7ef415e88f8cf0f9cae3252da6221376fd6d0c39012ba",
    "eng-rus": "fdee5ebdbe59c2a4293b875def3befdfddd990f95ecd10eb289eff22d815ed6d31267ca4695af66811ef368dd72b16342210475e044c995831e7931ce0e4bcfe",
}


def normalize(word):
    return unicodedata.normalize("NFC", word.casefold().replace("ё", "е").replace("\u0301", "")).strip()


def entries(folder, direction, stem):
    path = folder / (direction + ".tar.xz")
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha512").hexdigest() != SHA512[direction]:
            raise ValueError(f"Unexpected FreeDict source checksum for {direction}")
    pairs = defaultdict(dict)
    with tarfile.open(path) as archive:
        tei = next(member for member in archive if member.name.endswith(".tei"))
        with archive.extractfile(tei) as source:
            for _, entry in iterparse(source, events=("end",)):
                if not entry.tag.endswith("}entry"):
                    continue
                head = normalize(entry.findtext("./{*}form/{*}orth", ""))
                for item in entry.findall('.//{*}cit[@type="trans"]/{*}quote'):
                    value = normalize(item.text or "")
                    ru, en = (head, value) if direction == "rus-eng" else (value, head)
                    en = en.removeprefix("to ")
                    if re.fullmatch("[а-я]{3,}", ru) and re.fullmatch(r"[a-z]+(?:[ -][a-z]+){0,2}", en):
                        ru = stem(ru)
                        a, b = (ru, en) if direction == "rus-eng" else (en, ru)
                        pairs[a][b] = None
                entry.clear()
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "mailsearch/resources/ru_en.tsv.gz")
    args = parser.parse_args()
    stem = snowballstemmer.stemmer("russian").stemWord
    forward = entries(args.archives, "rus-eng", stem)
    reverse = entries(args.archives, "eng-rus", stem)
    with args.output.open("wb") as file, gzip.GzipFile(fileobj=file, filename="", mtime=0, mode="wb") as out:
        for ru, english in sorted(forward.items()):
            mutual = [en for en in english if ru in reverse.get(en, ())]
            if mutual:
                out.write((ru + "\t" + "|".join(mutual) + "\n").encode())


if __name__ == "__main__":
    main()
