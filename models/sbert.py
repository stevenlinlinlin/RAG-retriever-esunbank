from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import torch
import re

# SBERT model : paraphrase-multilingual-MiniLM-L12-v2, shibing624/text2vec-base-chinese
_model = None

def get_model():
    global _model
    if _model is None:
        # _model = SentenceTransformer('shibing624/text2vec-base-chinese')
        # _model = SentenceTransformer('lier007/xiaobu-embedding-v2') #最好的
        # _model = SentenceTransformer('iampanda/zpoint_large_embedding_zh') #
        # _model = SentenceTransformer('TencentBAC/Conan-embedding-v1')
        
        _model = SentenceTransformer('Pristinenlp/alime-embedding-large-zh')
        # _model = SentenceTransformer('dunzhang/stella-mrl-large-zh-v3.5-1792d')#('aspire/acge_text_embedding')#('Classical/Yinka')
    return _model
    # model = SentenceTransformer('lier007/xiaobu-embedding-v2')


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
    
    # FAISS
    # index = faiss.IndexFlatL2(query_vector.shape[1])
    # index.add(source_vectors)
    # distances, indices = index.search(query_vector, 1)
    # most_relevant_idx = indices[0][0]
    
    return source[most_relevant_idx]


############# doc split sentence #############
def embed_sentences(document):
    sentences = re.split(r'(。|！|\!|？|\?)', document)
    sentence_embeddings = np.array([embed_text_sbert(sentence) for sentence in sentences if sentence], dtype='float32')
    return sentences, sentence_embeddings

def faq_embed_sentences(document):
    sentences = re.split(r'(。|！|\!|？|\?)', document)
    # sentences = re.sub(r'[^a-zA-Z\u4e00-\u9fff]+', '', document)
    # document = document.replace(' ', '')
    # sentences = re.split(r'(?:。|！|\!|？|\?|\n)', document)
    sentence_embeddings = np.array([embed_text_sbert(sentence) for sentence in sentences if sentence], dtype='float32')
    return sentences, sentence_embeddings
    
def SBERT_retrieve_sentence(qs, source, corpus_dict, qs_category):
    query_vector = embed_text_sbert(qs).reshape(1, -1)
    
    best_reference = None
    best_sentence = None
    best_score = float('-inf')
    
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
        
        if max_similarity > best_score:
            best_score = max_similarity
            best_reference = i
            best_sentence = most_similar_sentence
    return source[best_reference]