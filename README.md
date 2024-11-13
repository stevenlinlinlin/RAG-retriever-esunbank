# RAG-retriever-esunbank
2024玉山人工智慧公開挑戰賽-RAG與LLM在金融問答的應用  
初賽：金融問答檢索找出相關文獻

## Leaderboard
初賽37名達前標25％（37/222）  
TEAM_6462 : 
[Leaderboard Link](https://tbrain.trendmicro.com.tw/Competitions/Details/37)  
![image](https://github.com/stevenlinlinlin/RAG-retriever-esunbank/blob/main/pictures/Leaderboard.png)

## Data
- 資料集在競賽官網有提供，分為三個主題：FAQ、Finance、Insurance
- 整理過後的資料集放在data資料夾中（google雲端）：https://drive.google.com/drive/folders/10pf7gvLdOFwsnBgTioXo_E93PHx0MgkQ?usp=sharing
### Preprocessing（Preprocess/data_preprocess.py）
1. 將競賽資料集（faq/finacne/insurance）讀取文字後存成txt檔
2. 針對finance中有些pdf是圖片檔(86個pdf)，使用OCR轉換成文字檔
3. 人工檢查這86個pdf轉換後的文字檔，將有問題的資料進行修正

## Model
- 主要使用Huggingface的transformers套件
- 利用上面有的models進行retriever
- SBERT: 'dunzhang/stella-mrl-large-zh-v3.5-1792d'
- Cross encoder model: 'BAAI/bge-reranker-large'
### Final method
- faq/insurance文檔
    - 一般的句子或文章
    - 切成chunks後進行SBERT相似度retriever
-  finance文檔
    - 表格或是有特殊格式
    - 只保留文字其他去除
    - 分別用BM25和Cross encoder model進行retriever
    - 透過給予不同權重進行ensemble（bm25_weight=0.2, cross_encoder_weight=0.8）

## Usage
### Environment
- OS : Ubuntu 20.04
- GPU : RTX4090
- python 3.9

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Download data
- 下載整理過後的資料集放在data資料夾中（google雲端）：https://drive.google.com/drive/folders/10pf7gvLdOFwsnBgTioXo_E93PHx0MgkQ?usp=sharing

### 3. Retrieval
```bash
python main_chunk_rerank.py \
    --question_path '../Test Dataset_Preliminary 1/questions_preliminary.json' \
    --source_path './data' \
    --output_path './predicts' \
    --model 'SBERT_sentence' \
```
- question_path: 測試資料集的問題
- source_path: 資料集的路徑
- output_path: 存放預測結果的路徑
