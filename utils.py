import os
import csv
import json
import pdfplumber
from tqdm import tqdm

def load_all_source_data(source_path, ):
    print("Loading insurance data...")
    source_path_insurance = os.path.join(source_path, 'insurance')
    corpus_dict_insurance = load_data(source_path_insurance)

    print("Loading finance data...")
    source_path_finance = os.path.join(source_path, 'finance')
    corpus_dict_finance = load_data(source_path_finance)

    print("Loading FAQ data...")
    key_to_source_dict = read_json(os.path.join(source_path, 'faq/pid_map_content.json'))
    key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}    
        
    return corpus_dict_insurance, corpus_dict_finance, key_to_source_dict

def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)

    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):
        text = page.extract_text()
        if text:
            pdf_text += text
    pdf.close()

    return pdf_text

def read_txt(txt_loc):
    with open(txt_loc, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# def load_q_source_data(q_source_list, source_path):
#     masked_file_ls = os.listdir(source_path)
#     corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in masked_file_ls if int(file.replace('.pdf', '')) in q_source_list}
#     return corpus_dict

def load_data(source_path):
    masked_file_ls = os.listdir(source_path)
    # masked_file_ls = [file for file in os.listdir(source_path) if file.endswith('.txt')]
    masked_file_ls = [file for file in os.listdir(source_path) if file.endswith('.txt')]

    # read pdf
    # corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}
    # read txt
    corpus_dict = {int(file.replace('.txt', '')): read_txt(os.path.join(source_path, file)) for file in masked_file_ls}
    return corpus_dict

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_model_results(output_path, file_name, ap_at_1):
    results_path = os.path.join(output_path, 'models_ap1.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["file_name", "AP@1"])
        writer.writerow([file_name, ap_at_1])
    return results_path

def write_answer_json(answer_dict, output_path, model_name):
    # answer_path = os.path.join(output_path, f"{model_name}_answers.json")
    answer_path = os.path.join(output_path, f"pred_retrieve.json")
    answer_path = get_unique_filename(answer_path)
    file_name_with_extension = os.path.basename(answer_path)
    file_name = os.path.splitext(file_name_with_extension)[0]
    
    with open(answer_path, 'w') as f:
        json.dump(answer_dict, f, indent=4)
    return answer_path, file_name

def get_unique_filename(answer_path):
    base, extension = os.path.splitext(answer_path)
    i = 1
    new_answer_path = answer_path
    # 檢查文件是否存在，若存在則加上 _{i} 直到不存在為止
    while os.path.exists(new_answer_path):
        new_answer_path = f"{base}_{i}{extension}"
        i += 1
    return new_answer_path

