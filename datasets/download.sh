#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/local"
msmarco_url="https://msmarco-gkbqakfehxgtezgg.b01.azurefd.net/msmarcoranking"

usage() {
  cat <<'USAGE'
Использование: ./datasets/download.sh НАБОР [НАБОР ...]
                ./datasets/download.sh --list

  rubq-2.0       Вопросы и параграфы Wikipedia, около 50 МБ
  miracl-ru     Русский корпус, запросы и qrels
  mr-tydi-ru    Русский корпус и размеченные запросы
  mmarco-ru     Русский корпус Google Translate, запросы и qrels
  msmarco-passage  Английские пассажи и запросы с разметкой
  msmarco-doc      Английские документы и запросы с разметкой
  beir-scifact    Небольшой англоязычный бенчмарк поиска
  lotte           Англоязычные форумы StackExchange
  enron           Публичный архив корпоративной почты, версия CMU 2015

Данные сохраняются в datasets/local/. Источники и условия описаны в datasets/README.md.
USAGE
}

download_hub() {
  local destination="$1" repository="$2" revision="$3"
  shift 3
  local hf_command=hf

  if [[ -x "$script_dir/../.venv/bin/hf" ]]; then
    hf_command="$script_dir/../.venv/bin/hf"
  elif ! command -v hf >/dev/null 2>&1; then
    printf '%s\n' 'Нужен Hugging Face CLI. Установи его командой uv tool install huggingface-hub.' >&2
    return 1
  fi

  "$hf_command" download "$repository" --repo-type dataset \
    --revision "$revision" --local-dir "$data_dir/$destination" "$@"
}

download_file() (
  local dataset="$1" filename="$2" url="$3"
  local destination="$data_dir/$dataset/$filename"

  if [[ -s "$destination" ]]; then
    printf 'Уже скачан: %s\n' "$destination"
    return 0
  fi

  mkdir -p -- "$(dirname -- "$destination")"
  local temporary
  temporary="$(mktemp "$destination.part.XXXXXXXX")"
  trap 'rm -f -- "$temporary"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  curl --fail --location --retry 3 --connect-timeout 15 --output "$temporary" "$url"
  if [[ ! -s "$temporary" ]]; then
    printf 'Пустой файл: %s\n' "$url" >&2
    return 1
  fi
  case "$filename" in
    *.gz) gzip -t -- "$temporary" ;;
    *.zip) unzip -tq "$temporary" ;;
  esac

  mv -f -- "$temporary" "$destination"
  printf 'Сохранён: %s\n' "$destination"
)

download_dataset() {
  local filename
  case "$1" in
    rubq-2.0)
      download_hub rubq-2.0/questions d0rj/RuBQ_2.0 main
      download_hub rubq-2.0/paragraphs d0rj/RuBQ_2.0-paragraphs 383cf3ba0395865ada07329c8601bee840cbeb2b
      ;;
    miracl-ru)
      download_hub miracl-ru/topics miracl/miracl 5be20db9509754dadad47689368639fcec739c00 \
        --include 'miracl-v1.0-ru/*' --include 'README.md'
      download_hub miracl-ru/corpus miracl/miracl-corpus d921ec7e349ce0d28daf30b2da9da5ee698bef0d \
        --include 'miracl-corpus-v1.0-ru/*' --include 'README.md'
      ;;
    mr-tydi-ru)
      download_hub mr-tydi-ru/topics castorini/mr-tydi 1d43c80218d06d0ef80f5b172ccabd848b948bc1 \
        --include 'mrtydi-v1.1-russian/*' --include 'README.md'
      download_hub mr-tydi-ru/corpus castorini/mr-tydi-corpus 3a3aa212bbe94a8cc0dc858710a3dad49d532054 \
        --include 'mrtydi-v1.1-russian/*' --include 'README.md'
      ;;
    mmarco-ru)
      download_hub mmarco-ru unicamp-dl/mmarco 6d039c4638c0ba3e46a9cb7b498b145e7edc6230 \
        --include 'data/google/collections/russian_collection.tsv' \
        --include 'data/google/queries/dev/russian_queries.dev.small.tsv' \
        --include 'data/google/queries/train/russian_queries.train.tsv' \
        --include 'data/qrels.dev.small.tsv' --include 'README.md'
      ;;
    msmarco-passage)
      download_file "$1" collectionandqueries.tar.gz "$msmarco_url/collectionandqueries.tar.gz"
      ;;
    msmarco-doc)
      for filename in msmarco-docs.tsv.gz msmarco-doctrain-queries.tsv.gz msmarco-doctrain-qrels.tsv.gz \
        msmarco-docdev-queries.tsv.gz msmarco-docdev-qrels.tsv.gz; do
        download_file "$1" "$filename" "$msmarco_url/$filename"
      done
      ;;
    beir-scifact)
      download_file "$1" scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
      ;;
    lotte)
      download_file "$1" lotte.tar.gz https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz
      ;;
    enron)
      download_file "$1" enron_mail_20150507.tar.gz https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
      ;;
  esac
}

if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h || "${1:-}" == --list ]]; then
  usage
  exit 0
fi

for dataset in "$@"; do
  case "$dataset" in
    rubq-2.0|miracl-ru|mr-tydi-ru|mmarco-ru|msmarco-passage|msmarco-doc|beir-scifact|lotte|enron) ;;
    *)
      printf 'Неизвестный набор: %s. Посмотри список через --list.\n' "$dataset" >&2
      exit 2
      ;;
  esac
done

for dataset in "$@"; do
  download_dataset "$dataset"
done
