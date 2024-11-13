from rank_bm25 import BM25Okapi
import jieba

def BM25_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # [TODO] 可自行替換其他檢索方式，以提升效能

    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut_for_search(qs))
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=1)
    a = ans[0]
    res = [key for key, value in corpus_dict.items() if value == a]
    return res[0]