import os
import argparse
from tqdm import tqdm

from utils import *
from models.bm25 import BM25_retrieve
from models.bert import BERT_retrieve
from models.sbert import SBERT_retrieve
from models.sbert_chunk import SBERT_retrieve_sentence
from models.sbert_chunk_rerank import SBERT_retrieve_chunk_rerank#, SBERT_retrieve_sentence_bm25#, SBERT_retrieve_sentence, SBERT_retrieve_sentence_bm25#, SBERT_retrieve_with_cross_encoder


def load_model(model_name):
    print(f"Loading model: {model_name}")
    if model_name == 'BM25':
        model = BM25_retrieve
    elif model_name == 'BERT':
        model = BERT_retrieve
    elif model_name == 'SBERT':
        model = SBERT_retrieve
    elif model_name == 'SBERT_sentence':
        finance_model = SBERT_retrieve_chunk_rerank #SBERT_retrieve_sentence #SBERT_retrieve_with_cross_encoder #
        model = SBERT_retrieve_sentence
    else:
        raise ValueError("Model not found")
    return model, finance_model


def main(corpus_dict_insurance, corpus_dict_finance, key_to_source_dict, question_path, model, finance_model):
    answer_dict = {"answers": []}

    qs_ref = read_json(question_path)
    
    for q_dict in tqdm(qs_ref['questions'], desc="Retrieving answers"):
        if q_dict['category'] == 'finance':
            retrieved = finance_model(q_dict['query'], q_dict['source'], corpus_dict_finance, q_dict['category'])
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            # continue
            retrieved = model(q_dict['query'], q_dict['source'], corpus_dict_insurance, q_dict['category'])
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            # continue
            corpus_dict_faq = {key: value for key, value in key_to_source_dict.items() if key in q_dict['source']}
            question_corpus_dict_faq = {}
            for key, value in corpus_dict_faq.items():
                questions = ''
                for v in value:
                    questions += v['question']
                question_corpus_dict_faq[key] = str(questions)
                
            retrieved = model(q_dict['query'], q_dict['source'], question_corpus_dict_faq, q_dict['category'])
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            # answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": 0})

        else:
            raise ValueError("Something went wrong") 
        
    return answer_dict


def evaluate(answer_dict, output_path, gt_path, model_name, eval=False):
    if eval:
        ground_truths = read_json(gt_path)
        gt_dict = {gt["qid"]: [gt["retrieve"], gt["category"]] for gt in ground_truths["ground_truths"]}

        # evaluate AP@1
        correct_count = 0
        incorrect_count_dict = {"finance": 0, "insurance": 0, "faq": 0}
        total_count = len(answer_dict["answers"])
        for ans in answer_dict["answers"]:
            qid = ans["qid"]
            if qid in gt_dict:
                if ans["retrieve"] == gt_dict[qid][0]:
                    correct_count += 1
                else:
                    incorrect_count_dict[gt_dict[qid][1]] += 1
                    print(f"Incorrect - qid: {qid}, Category: {gt_dict[qid][1]}, Retrieve answer: {ans['retrieve']}, GT answer: {gt_dict[qid][0]}")
            else:
                print(f"qid {qid} not found in ground truth")
        ap_at_1 = correct_count / total_count
        print(f"Incorrect count: {incorrect_count_dict}")
        print(f"AP@1: {ap_at_1:.4f}")
        # write results AP@1 to csv
        results_path = write_model_results(output_path, file_name, ap_at_1)
        print(f"Model results AP@1 written to {results_path}")
    
    os.makedirs(output_path, exist_ok=True)
    # write answers to json
    answer_path, file_name = write_answer_json(answer_dict, output_path, model_name)
    print(f"Answers written to {answer_path}")
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')
    parser.add_argument('--gt_path', type=str, required=False, help='ground truth path', default='../競賽資料集/dataset/preliminary/ground_truths_example.json')
    parser.add_argument('--model', type=str, required=False, help='模型名稱', default='BM25')
    parser.add_argument('--eval', type=bool, required=False, help='是否進行評估', default=False)

    args = parser.parse_args()
    model, finance_model = load_model(args.model)
    corpus_dict_insurance, corpus_dict_finance, key_to_source_dict = load_all_source_data(args.source_path) # 8mins
    answer_dict = main(corpus_dict_insurance, corpus_dict_finance, key_to_source_dict, args.question_path, model, finance_model)
    evaluate(answer_dict, args.output_path, args.gt_path, args.model, args.eval)
    
    