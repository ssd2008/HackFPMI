"""Fetch pinned model files on first setup and optionally prebuild the mail index."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from . import E5_MODEL, E5_REVISION, E5_QUERY_GRAPH, QWEN_MODEL, QWEN_REVISION
from .store import CACHE, DATA, load_jsonl, verify_manifest


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


def prepare_models(offline: bool):
    for name, (revision, files) in MODELS.items():
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


def main():
    parser = argparse.ArgumentParser(description="Подготовка моделей и поискового индекса")
    parser.add_argument("--offline", action="store_true", help="Работать только с локальным кешем моделей.")
    parser.add_argument("--models-only", action="store_true", help="Подготовить только файлы моделей.")
    parser.add_argument("--data", type=Path, default=DATA, help="Каталог с documents.jsonl и, при наличии, манифестом.")
    args = parser.parse_args()
    prepare_models(args.offline)
    if not args.models_only:
        from .dense import DenseIndex

        manifest = verify_manifest(args.data)
        documents = load_jsonl(args.data / "documents.jsonl")
        DenseIndex(documents, CACHE)
        if manifest.get("sha256"):
            print("Проверены контрольные суммы манифеста.")
        print(f"В индексе {len(documents)} писем.")


if __name__ == "__main__":
    main()
