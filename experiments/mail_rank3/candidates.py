"""Field-aware search and unsupervised subject/sender prototypes."""
from collections import defaultdict
from email.utils import parseaddr
from pathlib import Path
import hashlib
import re

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/"datasets/local/rank3"
ADDRESS=re.compile(r"[\w.+%-]+@([a-z0-9-]+(?:\.[a-z0-9-]+)+)",re.I)
# Only a complete sender-only residual can trigger exact field retrieval.
SENDER=re.compile(r"(?:(?:найди|покажи|выведи|отбери|мне нужны)\s+)?"
                  r"(?:все\s+)?(?:письма|письмо|сообщения|сообщение)\s+"
                  r"(?:с адресов на|с домена|от отправител(?:я|ей)(?: с домена)?|от)\s+"
                  r"(?P<address>[\w.+%-]*@[a-z0-9-]+(?:\.[a-z0-9-]+)+)"
                  r"(?:\s*[,;]?\s*полученн(?:ые|ое|ых))?[\s.?!]*",re.I)


DOMAIN_QUERY=re.compile(r"(?:(?:найди|покажи|выведи|отбери)\s+)?"
                        r"(?:письма|сообщения)\s*,?\s*(?:у которых\s+)?"
                        r"(?:домен отправителя|отправитель с домена)\s*[:=–—-]?\s*@?"
                        r"(?P<domain>[a-z0-9-]+(?:\.[a-z0-9-]+)+)[\s.?!]*",re.I)
MAIL_WORD=re.compile(r"^(?P<prefix>(?:(?:найди|покажи|выведи|отбери|мне нужны)\s+)?(?:все\s+)?)"
                     r"(?P<word>[а-яё]{5,10})(?=\s)",re.I)


def repair_mail_word(match):
    # Only normalize a single adjacent transposition in a structural noun.
    word=match["word"]
    for expected in ("письма","письмо","сообщения"):
        if len(word)==len(expected):
            for i in range(len(word)-1):
                if word[:i]+word[i+1]+word[i]+word[i+2:]==expected:
                    return match["prefix"]+expected
    return match[0]


def sender_parts(value):
    if not value:
        return "", ""
    addr=parseaddr(value)[1].casefold()
    match=ADDRESS.fullmatch(addr)
    if not match:
        return "", ""
    return addr,match[1]


class FieldIndex:
    def __init__(self,docs,store,parse_date,natural=False):
        self.natural=natural
        self.parse_date=parse_date
        self.postings=defaultdict(list)
        for d in docs:
            addr,domain=sender_parts(store.summaries[d["doc_id"]]["sender"])
            if domain:
                row=(d["doc_dt"] or "",d["doc_id"])
                for key in (addr,"@"+domain):
                    self.postings[d["synthetic_user"],key].append(row)
        for posting in self.postings.values():
            posting.sort(reverse=True)

    def find(self,box,query,at,limit):
        if "@" not in query and not (self.natural and "домен" in query.lower()):
            return None
        scope=self.parse_date(query,at)
        text=scope.text.strip()
        domain_match=None
        if self.natural:
            text=MAIL_WORD.sub(repair_mail_word,text)
            if scope.bounded:
                text=re.sub(r"\s+за(?:\s+период)?[\s.?!]*$","",text)
            domain_match=DOMAIN_QUERY.fullmatch(text)
        match=SENDER.fullmatch(text)
        if not match and not domain_match:
            return None
        upper=min(at,scope.end or at).isoformat()
        lower=scope.start.isoformat() if scope.start else ""
        result=[]
        key=match["address"] if match else "@"+domain_match["domain"]
        for day,did in self.postings.get((box,key.casefold()),()):
            if lower<=day<=upper and (day or not scope.bounded):
                result.append((did,1.0))
                if len(result)==limit:
                    break
        return result


def header_text(row):
    addr,domain=sender_parts(row["sender"])
    # Preserve the literal host and expose its component words to the vectorizer.
    return " ".join((row["subject"],row["subject"],addr,domain.replace("."," ")))


class HeaderModel:
    def __init__(self,docs,store,cache_dir=CACHE):
        texts=[header_text(store.summaries[d["doc_id"]]) for d in docs]
        digest=hashlib.sha256("\x00".join(texts).encode()).hexdigest()
        path=cache_dir/f"header-v1-{digest}.joblib"
        self.rows={d["doc_id"]:i for i,d in enumerate(docs)}
        if path.exists():
            self.vectorizer,self.matrix,self.svd,self.centers,self.labels=joblib.load(path)
            return
        self.vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,
                                       max_features=50000,sublinear_tf=True,dtype=np.float32)
        self.matrix=self.vectorizer.fit_transform(texts)
        self.svd=TruncatedSVD(n_components=48,n_iter=3,random_state=2026)
        latent=normalize(self.svd.fit_transform(self.matrix))
        clustering=MiniBatchKMeans(n_clusters=64,random_state=2026,batch_size=512,n_init=3)
        self.labels=clustering.fit_predict(latent)
        self.centers=normalize(clustering.cluster_centers_)
        tmp=path.with_suffix(".tmp")
        joblib.dump((self.vectorizer,self.matrix,self.svd,self.centers,self.labels),tmp)
        tmp.replace(path)

    def score(self,hits,text,method):
        idx=[self.rows[did] for did,_ in hits]
        q=self.vectorizer.transform([text])
        direct=(self.matrix[idx]@q.T).toarray().ravel()
        if method=="header":
            return direct
        latent=normalize(self.svd.transform(q))[0]
        return .6*direct+.4*(self.centers[self.labels[idx]]@latent)


class RoutedRanker:
    def __init__(self,base,field,header,method,parse_date):
        self.base,self.field,self.header,self.method,self.parse_date=base,field,header,method,parse_date

    def search(self,box,query,at,limit):
        if self.field:
            result=self.field.find(box,query,at,limit)
            if result is not None:
                return result
        if self.header is None:
            return self.base.search(box,query,at,limit)
        hits=self.base.search(box,query,at,max(32,limit))
        if len(hits)<2:
            return hits[:limit]
        scope=self.parse_date(query,at)
        signal=self.header.score(hits,scope.text,self.method)
        scores=[(did,value+.20*float(extra)) for (did,value),extra in zip(hits,signal)]
        return sorted(scores,key=lambda h:-h[1])[:limit]


def make(name,base,store,docs,parse_date):
    if name not in ("field","header","cluster","combined","field_header","natural"):
        raise ValueError(name)
    field=FieldIndex(docs,store,parse_date,natural=name=="natural") if name in ("field","combined","field_header","natural") else None
    header=None if name=="field" else HeaderModel(docs,store)
    method="header" if name in ("header","field_header") else "cluster"
    return RoutedRanker(base,field,header,method,parse_date)
