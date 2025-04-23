import os
import re
import math
import collections

import streamlit as st
from bs4 import BeautifulSoup
from pymorphy3 import MorphAnalyzer

morph = MorphAnalyzer()


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r'[\wа-яА-ЯёЁ]+', text.lower())
    return [morph.parse(tok)[0].normal_form for tok in raw_tokens]


def build_query_vector(query: str, idf: dict[str, float]) -> tuple[dict[str, float], float]:
    tokens = tokenize(query)
    if not tokens:
        return {}, 0.0

    tf_counts = collections.Counter(tokens)
    total = len(tokens)
    q_vec: dict[str, float] = {}
    for term, cnt in tf_counts.items():
        if term in idf:
            tf = cnt / total
            q_vec[term] = tf * idf[term]

    norm = math.sqrt(sum(v * v for v in q_vec.values()))
    return q_vec, norm


@st.cache_data(show_spinner=False)
def load_corpus():
    doc_vecs: dict[str, dict[str, float]] = {}
    doc_norms: dict[str, float] = {}
    idf: dict[str, float] = {}
    titles: dict[str, str] = {}

    tfidf_files = sorted(f for f in os.listdir('tfidf_lemmas') if f.endswith('.txt'))
    for fname in tfidf_files:
        page_num = re.search(r'(\d+)', fname).group(1)
        doc_name = f'page-{page_num}.html'
        doc_path = os.path.join('tfidf_lemmas', fname)

        vec: dict[str, float] = {}
        with open(doc_path, 'r', encoding='utf-8') as f:
            for line in f:
                term, idf_val, tfidf_val = line.strip().split()
                idf_val, tfidf_val = float(idf_val), float(tfidf_val)
                vec[term] = tfidf_val
                if term not in idf:
                    idf[term] = idf_val

        norm = math.sqrt(sum(v * v for v in vec.values()))
        doc_vecs[doc_name] = vec
        doc_norms[doc_name] = norm

        html_file = os.path.join('data', doc_name)
        if os.path.exists(html_file):
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as hf:
                soup = BeautifulSoup(hf, 'html.parser')
                title_tag = soup.title
                titles[doc_name] = title_tag.get_text(strip=True) if title_tag else doc_name
        else:
            titles[doc_name] = doc_name

    return doc_vecs, doc_norms, idf, titles


def search(query: str, k: int = 10) -> list[tuple[str, float]]:
    q_vec, q_norm = build_query_vector(query, IDF)
    if q_norm == 0:
        return []

    scores: list[tuple[str, float]] = []
    for doc, d_vec in DOC_VECS.items():
        dot = 0.0
        for term, q_w in q_vec.items():
            d_w = d_vec.get(term)
            if d_w is not None:
                dot += q_w * d_w
        if dot > 0:
            sim = dot / (q_norm * DOC_NORMS[doc])
            scores.append((doc, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

with st.spinner('Загрузка индекса…'):
    DOC_VECS, DOC_NORMS, IDF, TITLES = load_corpus()

st.title('Поисковая система')
query = st.text_input('Введите запрос')

if query:
    top_results = search(query)
    if not top_results:
        st.warning('Ничего не найдено')
    else:
        st.subheader('Результаты поиска:')
        for rank, (doc, score) in enumerate(top_results, 1):
            st.markdown(f'**{rank}. {TITLES.get(doc, doc)}**  ')
            st.caption(f'Релевантность: {score:.4f}')
            page_path = os.path.join('data', doc)
            if os.path.exists(page_path):
                with open(page_path, 'r', encoding='utf-8', errors='ignore') as fp:
                    raw_html = fp.read()
                with st.expander('Открыть страницу'):
                    st.components.v1.html(raw_html, height=400, scrolling=True)
