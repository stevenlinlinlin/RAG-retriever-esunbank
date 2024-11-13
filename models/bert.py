from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

# BERT models : bert-base-chinese, bert-base-multilingual-cased

_tokenizer = None
_model = None

def get_tokenizer_and_model(model_name="bert-base-chinese"):
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(model_name)
    if _model is None:
        _model = BertModel.from_pretrained(model_name)
    return _tokenizer, _model

def embed_text_bert(text):
    tokenizer, model = get_tokenizer_and_model()
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def BERT_retrieve(qs, source, corpus_dict):
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    
    source_vectors = np.array([embed_text_bert(doc) for doc in filtered_corpus], dtype='float32')
    query_vector = embed_text_bert(qs).reshape(1, -1)
    
    # Cosine Similarity
    # similarities = cosine_similarity(query_vector, source_vectors)
    # most_relevant_idx = np.argmax(similarities)
    
    # FAISS
    index = faiss.IndexFlatL2(query_vector.shape[1])
    index.add(source_vectors)
    distances, indices = index.search(query_vector, 1)
    most_relevant_idx = indices[0][0]
    
    return source[most_relevant_idx]