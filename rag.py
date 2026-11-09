"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answering, evaluation."""
import glob, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

def chunk_documents(folder="data/policies", size=40, overlap=10):
    chunks, step = [], size - overlap
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path), "text": " ".join(words[start:start + size])})
            if start + size >= len(words):      # this chunk reaches/passes the last word
                break
            start += step
    return chunks

class Retriever:
    def __init__(self, size=40, overlap=10):
        self.chunks = chunk_documents(size=size, overlap=overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])
    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in order]

_default = Retriever()
def retrieve(query, k=3):
    return _default.retrieve(query, k)

POLICY_SYSTEM = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
                 "If the answer is not in the context, say you could not find the information.")

def answer(question, retriever=None):
    hits = (retriever or _default).retrieve(question, 3)
    ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"
    r = mock_llm.chat([{"role": "system", "content": POLICY_SYSTEM}, {"role": "user", "content": user}])
    return r.choices[0].message.content, [h["source"] for h in hits]

def evaluate(retriever=None, verbose=True):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fb_ok = 0; failures = []
    for q in ev.itertuples():
        text, sources = answer(q.question, retriever)
        if q.expected_doc == "NONE":
            fb_ok += text.startswith("I could not find")
            if not text.startswith("I could not find"): failures.append((q.question, "no fallback", text))
        else:
            d = q.expected_doc in sources; f = q.expected_fact in text
            doc_ok += d; fact_ok += f
            if not (d and f): failures.append((q.question, f"doc_in_top3={d} fact_in_answer={f}", text))
    return {"doc_in_top3": doc_ok, "fact_in_answer": fact_ok, "fallback": fb_ok}, failures
