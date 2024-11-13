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
        _model = SentenceTransformer('dunzhang/stella-mrl-large-zh-v3.5-1792d', device='cuda')#('TencentBAC/Conan-embedding-v1')#最好的
        # print("Device:", _model.device)
        # print('torch', torch.cuda.is_available())
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


def faq_embed_sentences(document):
    sentences = re.split(r'(。|！|\!|？|\?)', document)
    sentences = [sentence for sentence in sentences]
    sentence_embeddings = embed_text_sbert(sentences)
    sentence_embeddings = np.array(sentence_embeddings, dtype='float32')
    
    return sentences, sentence_embeddings

def embed_sentences(document):
    document = document.replace(' ', '')  # 移除空格
    # 定義切割的字數
    segment_length = 256
    overlap = 200
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