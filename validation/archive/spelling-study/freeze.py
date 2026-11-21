"""Freeze one edit per public query. Does not access the corpus or relevance labels."""
import hashlib, json, re
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SKIP = re.compile(r"^(?:покаж|найд|отыщ|письм|сообщени|уведомлени|полученн|вчера|сегодн)")
TOKEN = re.compile(r'(?<![\w@-])[а-яё]{5,}(?![\w@-])')
# Use adjacent Cyrillic-key typo characters in the same keyboard row.
ROWS = ("йцукенгшщзхъ","фывапролджэ","ячсмитьбю")
def diagnostic(q):
    text = q['query']
    region = text.split(',', 1)[0]
    choices = [m for m in TOKEN.finditer(region) if not SKIP.match(m[0])]
    if not choices:
        raise ValueError(text)
    data = hashlib.sha256(('mail-typo-v1/'+text).encode()).digest()
    match = choices[data[0] % len(choices)]
    word = match[0]
    op = ('transpose','replace','delete','insert')[data[1] % 4]
    i = 1 + data[2] % (len(word)-2)
    if op == 'transpose':
        possible = [k for k in range(1,len(word)-1) if word[k] != word[k+1]]
        i = possible[data[2] % len(possible)]
        bad = word[:i] + word[i+1] + word[i] + word[i+2:]
    elif op == 'delete':
        bad = word[:i] + word[i+1:]
    elif op == 'insert':
        bad = word[:i] + word[i] + word[i:]
    else:
        row = next((r for r in ROWS if word[i] in r), ROWS[0])
        pos = row.find(word[i])
        neighbour = row[pos+1] if pos+1 < len(row) else row[pos-1]
        bad = word[:i]+neighbour+word[i+1:]
    assert bad != word
    return {**q,'query':text[:match.start()]+bad+text[match.end():], 'clean_query':text,
            'edit':op,'correct_word':word,'typed_word':bad}
src=ROOT/'data/queries_dev.jsonl'
rows=[diagnostic(q) for q in map(json.loads,src.open())]
out=HERE/'typos99.jsonl'
raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
if out.exists():
    assert out.read_bytes()==raw
else:
    out.write_bytes(raw)
print('frozen',len(rows),hashlib.sha256(raw).hexdigest(),Counter(r['edit'] for r in rows))
print('unique',len({r['query'] for r in rows}),'spelling pairs',len({(r['correct_word'],r['typed_word']) for r in rows}))
