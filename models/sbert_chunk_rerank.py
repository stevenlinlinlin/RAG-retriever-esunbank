from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import torch
import re
from rank_bm25 import BM25Okapi
import jieba
from FlagEmbedding import FlagReranker
cross_encoder_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True, device='cuda') # Setting use_fp16 to True speeds up computation with a slight  
# SBERT model : paraphrase-multilingual-MiniLM-L12-v2, shibing624/text2vec-base-chinese
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('dunzhang/stella-mrl-large-zh-v3.5-1792d', device='cuda')#('TencentBAC/Conan-embedding-v1')#最好的
        # print("Device:", _model.device)
        # print('torch', torch.cuda.is_available())
    return _model


def embed_text_sbert(text):
    model = get_model()
    return model.encode(text)

def SBERT_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    source_vectors = np.array([embed_text_sbert(doc) for doc in filtered_corpus], dtype='float32')
    # print(source_vectors.shape)
    query_vector = embed_text_sbert(qs).reshape(1, -1)
    
    # Cosine Similarity
    similarities = cosine_similarity(query_vector, source_vectors)
    most_relevant_idx = np.argmax(similarities)
    return source[most_relevant_idx]


def faq_embed_sentences(document):
    sentences = re.split(r'(。|！|\!|？|\?)', document)
    sentences = [sentence for sentence in sentences]
    sentence_embeddings = embed_text_sbert(sentences)
    sentence_embeddings = np.array(sentence_embeddings, dtype='float32')
    
    return sentences, sentence_embeddings

def embed_sentences(document):
    document = document.replace(' ', '')
    # 定義切割的字數
    segment_length = 256
    overlap = 200
    # segment_length = 20
    # overlap = 10
    step = segment_length - overlap
    
    if len(document) <= segment_length:
        segments = [document]  # 若 document 長度小於 segment_length，直接返回整段文本
    else:    
        # 循環切割文本
        segments = [document[i:i + segment_length] for i in range(0, len(document), step) if i + segment_length <= len(document)]

    
    # 嵌入每個切片
    # segment_embeddings = np.array([embed_text_sbert(segment) for segment in segments], dtype='float32')
    segments = [segment for segment in segments] # if segment
    segment_embeddings = embed_text_sbert(segments)
    segment_embeddings = np.array(segment_embeddings, dtype='float32')
    
    return segments, segment_embeddings

# retrieve
def SBERT_retrieve_sentence_og(qs, source, corpus_dict, qs_category):

    query_vector = embed_text_sbert(qs).reshape(1, -1)
    
    best_reference = None
    best_sentence = None
    best_score = float('-inf')
    
    top_docs = []
    
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    for i, doc in enumerate(filtered_corpus):
        if not doc:
            continue
        if qs_category != 'finance': 
            sentences, sentence_vectors = faq_embed_sentences(doc)
        else:
            sentences, sentence_vectors = embed_sentences(doc)
        similarities = cosine_similarity(query_vector, sentence_vectors)
        max_similarity = similarities.max()
        max_index = similarities.argmax()
        most_similar_sentence = sentences[max_index]
        
        top_docs.append({
            'doc_index': i,
            'max_similarity': max_similarity,
            'most_similar_sentence': most_similar_sentence
        })
    top_docs = sorted(top_docs, key=lambda x: x['max_similarity'], reverse=True)[:5]
    candidates_source = [source[candidate['doc_index']] for candidate in top_docs]
    return BM25_retrieve(qs, candidates_source, corpus_dict)

# SBERT -> BM25
def SBERT_bm25_retrieve_sentence_og(qs, source, corpus_dict):
    query_vector = embed_text_sbert(qs).reshape(1, -1)

    top_docs = []

    filtered_corpus = [corpus_dict[int(file)] for file in source]
    for i, doc in enumerate(filtered_corpus):
        if not doc:
            continue
        sentences, sentence_vectors = embed_sentences(doc)
        similarities = cosine_similarity(query_vector, sentence_vectors)
        max_similarity = similarities.max()
        max_index = similarities.argmax()
        most_similar_sentence = sentences[max_index]
        
        top_docs.append({
            'doc_index': i,
            'max_similarity': max_similarity,
            'most_similar_sentence': most_similar_sentence
        })
    
    # Top 10 most similar documents
    top_docs = sorted(top_docs, key=lambda x: x['max_similarity'], reverse=True)[:50]
    candidates_source = [source[candidate['doc_index']] for candidate in top_docs]
    
    return BM25_retrieve(qs, candidates_source, corpus_dict)

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


def SBERT_retrieve_chunk_rerank(qs, source, corpus_dict, qs_category, bm25_weight=0.2, cross_encoder_weight=0.8):
    query_vector = embed_text_sbert(qs).reshape(1, -1)
    
    # filtered_candidates = [corpus_dict[int(file)] for file in source]
    filtered_candidates = [re.sub(r'[^a-zA-Z\u4e00-\u9fff]+', '', corpus_dict[int(file)]) for file in source]

    # # BM25 檢索候選結果
    # filtered_candidates = [corpus_dict[int(file)] for file in candidates_source]
    tokenized_candidates = [list(jieba.cut_for_search(doc)) for doc in filtered_candidates]
    bm25 = BM25Okapi(tokenized_candidates)
    tokenized_query = list(jieba.cut_for_search(qs))
    bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=np.float64).flatten()

    # Cross-Encoder 重新排序
    cross_encoder_inputs = [(qs, candidate) for candidate in filtered_candidates]
    cross_encoder_scores = np.array(cross_encoder_model.compute_score(cross_encoder_inputs), dtype=np.float64).flatten()

    # 確保 bm25_scores 和 cross_encoder_scores 都是浮點數並且數量一致
    if len(bm25_scores) != len(cross_encoder_scores):
        raise ValueError("BM25 與 Cross-Encoder 的分數數量不一致。")
    # print(sbert_score)
    # print(bm25_scores)
    # print(cross_encoder_scores)

    # 計算加權平均分數
    weighted_scores = [
        bm25_weight * bm25_scores[i] + cross_encoder_weight * cross_encoder_scores[i]
        for i in range(len(bm25_scores))
    ]

    # 找出加權總分最高的文本 ID
    best_idx = int(np.argmax(weighted_scores))
    # best_key = candidates_source[best_idx]
    best_key = source[best_idx]
    best_score = weighted_scores[best_idx]
    # print("最高分的加權分數:", best_score)
    
    return best_key