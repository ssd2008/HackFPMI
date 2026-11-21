from hashlib import file_digest
from pathlib import Path
import argparse
import json
import re


def read(path):
    return json.loads(path.read_bytes())


def file_hash(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=('small', 'large'))
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = read(args.output)
    previous = read(Path('results') / f'ood_transfer_{args.group}_candidate.json')
    previous_dir = Path('.cache') / f'ood-transfer-{args.group}-candidate'
    cache = args.cache
    protocol = out['protocol']; prev = previous['protocol']
    assert file_hash(cache / 'freeze.json') == out['frozen_sha256']
    assert protocol == read(cache / 'freeze.json')
    assert protocol['inputs'] == prev['inputs']
    assert protocol['engines']['baseline'] == prev['engines']['current']
    assert not re.search(r'/Users/|ranked_ids|[\w.+-]+@[\w.-]+\.(?:com|org|net|ru)\b', args.output.read_text(), re.I)
    assert set(protocol['inputs']) == set(out['datasets']) == set(previous['datasets'])
    result = []
    for ds, section in out['datasets'].items():
        prior = previous['datasets'][ds]
        before, after = protocol['engines'].values()
        provenance = section['ranking_provenance']
        assert provenance['baseline']['source_freeze_sha256'] == previous['frozen_sha256']
        assert provenance['baseline']['source_artifact_sha256'] == file_hash(previous_dir / ds / (before + '.json'))
        assert provenance['current']['artifact_sha256'] == file_hash(cache / ds / (after + '.json'))
        assert section['passage_vectors_sha256'] == prior['passage_vectors_sha256']
        assert provenance['current']['status'] == 'computed' and provenance['baseline']['status'] == 'reused'
        for ch, scores in section['channels'].items():
            prior_scores = prior['channels'][ch]['current']
            b, a = scores['baseline'], scores['current']
            assert b == prior_scores and a['mailbox_or_asof_violations'] == 0
            assert scores['queries'] == protocol['inputs'][ds]['queries'][ch]
            assert sum(scores['ranked_lists'].values()) == scores['queries']
            ks=['ndcg_at_10', 'known_recall_at_10', 'hit_at_10', 'mrr_at_10', 'false_nonempty_rate']
            result.append([ds, ch, scores['queries'], scores['ranked_lists']['different'], [round(a[k]-b[k], 8) for k in ks if k in a], a['returned_documents_checked']])
    print('\n'.join(' '.join(map(str,x)) for x in result))


if __name__ == '__main__':
    main()
