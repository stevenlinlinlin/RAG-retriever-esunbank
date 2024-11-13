#!/bin/bash

# python main_chunk_rerank.py \
#     --question_path '../競賽資料集/dataset/preliminary/questions_example.json' \
#     --source_path './data' \
#     --output_path './results' \
#     --gt_path '../競賽資料集/dataset/preliminary/ground_truths_example.json' \
#     --model 'SBERT_sentence' \
#     --eval \

python main_chunk_rerank.py \
    --question_path '../Test Dataset_Preliminary 1/questions_preliminary.json' \
    --source_path './data' \
    --output_path './predicts' \
    --model 'SBERT_sentence' \