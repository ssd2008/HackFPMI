# Как разметить поиск в своей почте

На новой машине подготовьте два каталога с `documents.jsonl`, по одному на владельца. В первый включите все нужные ящики владельца dev, во второй — все ящики владельца blind. Подставьте свои пути и запустите из корня проекта в Bash или Zsh.

```bash
judge_dev_data='/путь/к/снимку-dev'
judge_blind_data='/путь/к/снимку-blind'
judge_root=.cache/validation-owners

python3 tools/prepare_judging.py init --dev-data "$judge_dev_data" \
  --blind-data "$judge_blind_data" --output "$judge_root"
```

В `dev/tasks.jsonl` и `blind/tasks.jsonl` появятся по шесть пустых задач. Первый владелец получает обозначение `owner-01`, второй — `owner-02`. Инструмент сохраняет идентификаторы ящиков и контрольные суммы в локальный `owners.json`. Для уже созданного пилота сразу откройте свои бланки. Если технические проверки на этих архивах повлияли на поиск, укажите, что оцениваете новые задачи знакомых владельцев.

1. **Запишите задачи по памяти до просмотра выдач.** В каждой строке укажите `query`, ящик `synthetic_user` из реестра, календарную `query_date`, цель `goal` и критерий `success_criterion`. Запишите исходное намерение в `task_type`. Для поиска конкретного письма используйте `known_item`, для подборки — `collection`, для проверки отсутствия — `empty`. У вариантов одной задачи укажите общий `family_id`, название `variant` и ссылку `original_query_id` на исходную формулировку. У отдельной исходной задачи без `family_id` её `query_id` станет идентификатором группы.
2. **Независимо подтвердите известные цели.** После записи запросов владелец может сам найти письмо в переписке и внести его ID в `known_doc_ids`. Поле `manual_candidate_ids` принимает дополнительные письма, в том числе правдоподобные ошибки. Оба списка допускается оставить пустыми. Владелец закрытой части хранит её вопросы и оценки у себя до финального сравнения.
3. **Соберите пул.** Выгрузите запросы и прогоните их минимум через две разные заранее зафиксированные версии на одном снимке. Команда `pool` объединяет первые десять писем из каждой выдачи с добавленными вручную, убирает повторы и перемешивает кандидатов. В копии для разметчика видны запрос, критерий и содержимое письма. Поле `relevance` у каждого кандидата пустое. У всех писем проверяются ящик и доступность на дату запроса.
4. **Проставьте оценки в копии пула.** Оценка `2` означает целевое письмо или прямое доказательство, `1` — полезный контекст, `0` — просмотренное нерелевантное письмо. Для ещё не просмотренного письма оставьте `null`. Если доступ разрешён, дайте второму человеку отдельно разметить примерно 20% задач, сравните оценки и согласуйте расхождения. Команда `qrels` принимает одну итоговую копию, сверяет её с исходным пулом и экспортирует явные оценки.
5. **Зафиксируйте исход в `tasks.jsonl`.** В поле `state` поставьте `positive`, когда есть положительные qrels. Для `known_item` оставьте в `known_doc_ids` все допустимые целевые письма, каждое с оценкой 1 или 2. Если владелец отдельно проверил отсутствие совпадений в своём ящике на заданную дату, укажите `verified_empty` и кратко запишите в `empty_verification`, как он это проверил. Если положительных оценок и подтверждения отсутствия пока нет, укажите `unjudged`. Из пустой выдачи вывод об отсутствии письма не следует.

## Команды для одного владельца

Для закрытой части используйте `blind`, владельца `owner-02` и каталог из `judge_blind_data`. Результаты разметки содержат полные письма и хранятся локально.

```bash
judge_root=.cache/validation-owners
judge_dir="$judge_root/dev"
judge_data="$judge_dev_data"
judge_owner=owner-01
judge_args=(--owners "$judge_root/owners.json" --owner "$judge_owner"
            --tasks "$judge_dir/tasks.jsonl" --data "$judge_data")

python3 tools/prepare_judging.py queries "${judge_args[@]}" \
  --output "$judge_dir/queries.jsonl"
```

Для уже запущенных сравниваемых версий укажите их локальные адреса. В примере они используют порты 18780 и 18781.

```bash
.venv/bin/python benchmark.py --data "$judge_data" --queries "$judge_dir/queries.jsonl" \
  --unjudged --k 10 --repeats 1 --url http://127.0.0.1:18780 --output "$judge_dir/run-a.json"
.venv/bin/python benchmark.py --data "$judge_data" --queries "$judge_dir/queries.jsonl" \
  --unjudged --k 10 --repeats 1 --url http://127.0.0.1:18781 --output "$judge_dir/run-b.json"

python3 tools/prepare_judging.py pool "${judge_args[@]}" \
  --predictions "$judge_dir/run-a.json" "$judge_dir/run-b.json" \
  --queries-file "$judge_dir/queries.jsonl" --output "$judge_dir/pool.json"

cp -n "$judge_dir/pool.json" "$judge_dir/consensus.json"
```

Заполните `relevance` в `consensus.json`, затем выполните команду.

```bash
python3 tools/prepare_judging.py qrels "${judge_args[@]}" \
  --pool "$judge_dir/pool.json" --consensus "$judge_dir/consensus.json" \
  --output "$judge_dir/qrels.jsonl"
```

После заполнения `state` у каждой задачи создайте метаданные и сравните те же сохранённые выдачи. Команда `metadata` сверяет состояния и целевые письма с qrels. В `source` автоматически записывается обезличенное обозначение владельца.

```bash
python3 tools/prepare_judging.py metadata "${judge_args[@]}" \
  --qrels "$judge_dir/qrels.jsonl" --output "$judge_dir/metadata.jsonl"

.venv/bin/python tools/validate_rankings.py \
  --before "$judge_dir/run-a.json" --after "$judge_dir/run-b.json" \
  --queries "$judge_dir/queries.jsonl" --qrels "$judge_dir/qrels.jsonl" \
  --tasks "$judge_dir/metadata.jsonl" --output "$judge_dir/paired.json"
```

Для широких задач можно заранее запросить `--k 20` у всех версий и указать `--depth 20` при подготовке пула. Для запросов, пула, qrels и метаданных задавайте новый путь вывода. Инструмент подготовки сохраняет существующие файлы разметки.
