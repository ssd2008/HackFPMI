"""Audit the only changed production function on the actual supplied query stream."""
from collections import defaultdict
from dataclasses import astuple
from datetime import date
from hashlib import file_digest, sha256
from pathlib import Path
from types import ModuleType
import io
import json
import subprocess
import sys
import tarfile

import numpy as np

OLD, NEW = sys.argv[1:3]
ROOT = Path.cwd()
BASE = ROOT / '.cache/round4'
BENCH = Path.home() / 'Downloads/HackFPMI-mail-OOD/external/benchmarks'


def git(*args):
    return subprocess.check_output(['git', *args])


def filehash(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()


def lines(path):
    with path.open('rb') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dates(commit):
    name = 'audit_dates_' + commit
    m = ModuleType(name)
    sys.modules[name] = m
    source = git('show', commit + ':mailsearch/dates.py')
    exec(compile(source, name, 'exec'), m.__dict__)
    return m.parse_date_scope


class Parsed(Exception):
    pass


def main():
    old = git('rev-parse', OLD + '^{commit}').decode().strip()
    new = git('rev-parse', NEW + '^{commit}').decode().strip()
    if git('diff', '--name-only', old, new, '--', 'mailsearch').decode().splitlines() != ['mailsearch/dates.py']:
        raise ValueError('The date parser must be the only production file that changed')
    package = git('archive', old, 'mailsearch')
    snapshot = BASE / 'date-audit-source' / old
    snapshot.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(package)) as archive:
        archive.extractall(snapshot, filter='data')
    sys.path.insert(0, str(snapshot))
    import mailsearch.retrieval as retrieval
    from mailsearch.fields import MailFields
    from mailsearch.spelling import CorpusSpelling

    before, after = load_dates(old), load_dates(new)
    result = {'before': old, 'after': new, 'production_files_changed': ['mailsearch/dates.py'],
              'method': 'Compare full (remaining query text, start, end) for every raw question and every actual date-parser call from the pinned production search method. The latter is intercepted after mail-field extraction, quote removal, corpus-derived spelling correction and article extraction. Early exits bypass the changed code. No relevance judgments are used.',
              'datasets': {}}
    for group in ('small', 'large'):
        frozen = BASE / ('transfer-' + group) / 'freeze.json'
        protocol = json.loads(frozen.read_bytes())
        if protocol['engines']['current'] != old:
            raise ValueError('Wrong source ranking freeze')
        for name, data in protocol['inputs'].items():
            folder = BENCH / name
            checked = ['documents.jsonl'] + [f'queries_{ch}.jsonl' for ch in data['queries']]
            if any(filehash(folder / f) != data['sha256'][f] for f in checked):
                raise ValueError('Changed input')
            docs = lines(folder / 'documents.jsonl')
            engine = retrieval.SearchEngine.__new__(retrieval.SearchEngine)
            # Only fields read before a call to parse_date_scope are required.
            boxes = defaultdict(list)
            for i, doc in enumerate(docs):
                boxes[doc['synthetic_user']].append(i)
            engine.boxes = {b: np.asarray(rows, np.int32) for b, rows in boxes.items()}
            engine.by_id = {doc['doc_id']: i for i, doc in enumerate(docs)}
            engine.ids = [doc['doc_id'] for doc in docs]
            engine.dates = np.array([date.fromisoformat(doc['doc_dt']).toordinal() if doc['doc_dt'] else 0 for doc in docs], np.int32)
            engine.fields = MailFields(docs)
            engine.spelling = CorpusSpelling((d['doc'] for d in docs), retrieval._stem)

            stats = {'queries': 0, 'raw_different': 0, 'parser_calls': 0, 'effective_different': 0, 'early_exits': 0}
            material = [sha256(), sha256()]

            def observe(text, ref):
                left, right = astuple(before(text, ref)), astuple(after(text, ref))
                stats['parser_calls'] += 1
                stats['effective_different'] += left != right
                for digest, row in zip(material, (left, right)):
                    digest.update((json.dumps([qid, *row], default=str, ensure_ascii=True) + '\n').encode())
                raise Parsed

            retrieval.parse_date_scope = observe
            for channel, count in data['queries'].items():
                rows = lines(folder / f'queries_{channel}.jsonl')
                assert len(rows) == count
                for q in rows:
                    query, ref, qid = q['query'], date.fromisoformat(q['query_date']), q['query_id']
                    stats['queries'] += 1
                    stats['raw_different'] += astuple(before(query, ref)) != astuple(after(query, ref))
                    try:
                        engine.search(q['synthetic_user'], query, ref, 10)
                    except Parsed:
                        continue
                    stats['early_exits'] += 1
            assert stats['early_exits'] + stats['parser_calls'] == stats['queries']
            stats['before_parser_result_sha256'], stats['after_parser_result_sha256'] = [d.hexdigest() for d in material]
            stats['ranking_freeze_sha256'] = filehash(frozen)
            result['datasets'][name] = stats
            print(name, stats['queries'], stats['parser_calls'], stats['raw_different'], stats['effective_different'], flush=True)
            del engine, boxes, docs
    output = BASE / ('date-equivalence-' + old[:7] + '-' + new[:7] + '.json')
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
