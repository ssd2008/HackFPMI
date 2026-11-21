import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def load_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


MONTHS = {
    name: i + 1
    for i, name in enumerate(
        'января февраля марта апреля мая июня июля августа '
        'сентября октября ноября декабря'.split()
    )
}
STOP = set(
    'покажи найди письма письмо уведомления уведомление полученные '
    'вчера о об с для на и в во по к от за до из при или же это '
    'как что не а но у со года'.split()
) | set(MONTHS)


def tokenize(text):
    tokens = re.findall(r'[a-zа-я0-9]+', text.lower().replace('ё', 'е'))
    return [token for token in tokens if token not in STOP]


def received_day(query):
    text = query['query'].lower()
    if re.search(r'полученные вчера', text):
        anchor = dt.date.fromisoformat(query['query_date'])
        return (anchor - dt.timedelta(days=1)).isoformat()
    match = re.search(
        r'полученные (\d{1,2}) ([а-яё]+) (\d{4}) года', text
    )
    if match:
        return dt.date(
            int(match[3]), MONTHS[match[2]], int(match[1])
        ).isoformat()
    raise ValueError('Дата не распознана диагностическим парсером')


def evaluate(order, positives, ids):
    if not positives:
        raise ValueError('Для пустых qrels нужна отдельная политика оценки')
    hits = np.array([ids[i] in positives for i in order], dtype=float)
    out = {
        f'Recall@{k}': float(hits[:k].sum() / len(positives))
        for k in (5, 10, 20)
    }
    discount = 1 / np.log2(np.arange(2, 12))
    ideal = discount[:min(10, len(positives))].sum()
    top = hits[:10]
    out['nDCG@10'] = float((top * discount[:len(top)]).sum() / ideal)
    found = np.flatnonzero(top)
    out['MRR@10'] = float(1 / (found[0] + 1)) if len(found) else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    args = parser.parse_args()
    documents = load_jsonl(args.data_dir / 'documents.jsonl')
    queries = load_jsonl(args.data_dir / 'queries_dev.jsonl')
    qrels = load_jsonl(args.data_dir / 'qrels_dev.jsonl')

    positives = collections.defaultdict(set)
    for pair in qrels:
        if pair['relevance'] > 0:
            positives[pair['query_id']].add(pair['doc_id'])

    texts = [doc['doc'] for doc in documents]
    word_vectorizer = CountVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False
    )
    counts = word_vectorizer.fit_transform(texts).tocsr().astype(float)
    n = counts.shape[0]
    frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (n - frequency + 0.5) / (frequency + 0.5))
    length = np.asarray(counts.sum(axis=1)).ravel()
    k1, b = 1.2, 0.75
    bm25_matrix = counts.copy()
    row = np.repeat(np.arange(n), np.diff(bm25_matrix.indptr))
    tf = bm25_matrix.data
    bm25_matrix.data = (
        idf[bm25_matrix.indices] * tf * (k1 + 1)
        / (tf + k1 * (1 - b + b * length[row] / length.mean()))
    )
    word_queries = word_vectorizer.transform([q['query'] for q in queries])
    word_queries.data[:] = 1
    bm25_scores = (bm25_matrix @ word_queries.T).toarray().T

    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5), min_df=2,
        max_features=160000, sublinear_tf=True,
    )
    char_documents = char_vectorizer.fit_transform(texts)
    char_queries = char_vectorizer.transform([q['query'] for q in queries])
    char_scores = (char_queries @ char_documents.T).toarray()

    ids = np.array([doc['doc_id'] for doc in documents])
    users = np.array([doc['synthetic_user'] for doc in documents])
    dates = np.array([doc['doc_dt'] or '' for doc in documents])
    methods = ('date_only', 'bm25', 'bm25_date', 'char_tfidf', 'char_tfidf_date')
    all_scores = {name: [] for name in methods}
    single_scores = {name: [] for name in methods}
    candidate_counts = []
    exact_date_sets = 0

    for qi, query in enumerate(queries):
        day = received_day(query)
        eligible = (
            (users == query['synthetic_user'])
            & ((dates == '') | (dates <= query['query_date']))
        )
        candidates = np.flatnonzero(eligible)
        daily = np.flatnonzero(eligible & (dates == day))
        relevant = positives[query['query_id']]
        candidate_counts.append((len(candidates), len(daily), len(relevant)))
        exact_date_sets += int(set(ids[daily]) == relevant)

        def ranked(candidates_, scores_):
            return candidates_[np.argsort(-scores_[candidates_], kind='stable')]

        orders = {
            'date_only': daily[np.argsort(ids[daily], kind='stable')],
            'bm25': ranked(candidates, bm25_scores[qi]),
            'bm25_date': ranked(daily, bm25_scores[qi]),
            'char_tfidf': ranked(candidates, char_scores[qi]),
            'char_tfidf_date': ranked(daily, char_scores[qi]),
        }
        for name, order in orders.items():
            result = evaluate(order, relevant, ids)
            all_scores[name].append(result)
            if len(relevant) == 1:
                single_scores[name].append(result['MRR@10'])

    report = {}
    for name in methods:
        report[name] = {
            metric: float(np.mean([row[metric] for row in all_scores[name]]))
            for metric in ('Recall@5', 'Recall@10', 'Recall@20', 'nDCG@10')
        }
        report[name]['MRR@10_one_relevant'] = float(np.mean(single_scores[name]))
    report['recall_ceilings'] = {
        str(k): float(np.mean([
            min(k, len(positives[q['query_id']])) / len(positives[q['query_id']])
            for q in queries
        ]))
        for k in (5, 10, 20, 50)
    }
    report['exact_date_sets'] = exact_date_sets
    report['perfect_char_date_ndcg'] = int(np.isclose(
        [row['nDCG@10'] for row in all_scores['char_tfidf_date']],
        1.0, rtol=0.0, atol=1e-12,
    ).sum())
    count_array = np.asarray(candidate_counts)
    report['candidate_counts'] = {
        name: {
            'min': int(count_array[:, i].min()),
            'median': float(np.median(count_array[:, i])),
            'mean': float(count_array[:, i].mean()),
            'max': int(count_array[:, i].max()),
        }
        for i, name in enumerate(('eligible', 'same_day', 'positives'))
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
