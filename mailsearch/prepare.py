"""Fetch pinned model files on first setup and optionally prebuild the mail index."""

import argparse
import hashlib
import json

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from . import E5_MODEL, E5_QUERY_GRAPH, E5_REVISION, QWEN_MODEL, QWEN_REVISION
from .store import CACHE, DATA, load_jsonl


MODELS = {
    E5_MODEL: (
        E5_REVISION,
        ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "sentencepiece.bpe.model", E5_QUERY_GRAPH),
    ),
    QWEN_MODEL: (
        QWEN_REVISION,
        ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "merges.txt", "vocab.json"),
    ),
}


def prepare_models(offline: bool, search_only: bool = False):
    for name, (revision, files) in MODELS.items():
        if search_only and name == QWEN_MODEL:
            continue
        for filename in files:
            options = dict(repo_id=name, filename=filename, revision=revision)
            try:
                hf_hub_download(**options, local_files_only=True)
            except LocalEntryNotFoundError:
                if offline:
                    raise SystemExit(f"В локальном кеше отсутствует {name}/{filename}. "
                                     "Запустите подготовку без --offline.")
                print(f"Загрузка {name}/{filename}…", flush=True)
                hf_hub_download(**options)
        print(f"Модель готова: {name}", flush=True)


def verify_data():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["sha256"].items():
        with (DATA / filename).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"Неверная контрольная сумма у {filename}")


def main():
    parser = argparse.ArgumentParser(description="Подготовка моделей и поискового индекса")
    parser.add_argument("--offline", action="store_true", help="Работать только с локальным кешем моделей.")
    parser.add_argument("--models-only", action="store_true", help="Подготовить только файлы моделей.")
    parser.add_argument("--search-only", action="store_true", help="Подготовить только поисковую модель.")
    args = parser.parse_args()
    prepare_models(args.offline, args.search_only)
    if not args.models_only:
        from .dense import DenseIndex

        verify_data()
        documents = load_jsonl(DATA / "documents.jsonl")
        DenseIndex(documents, CACHE)
        print(f"Проверены контрольные суммы. В индексе {len(documents)} писем.")


if __name__ == "__main__":
    main()
