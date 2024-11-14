import os
import pdfplumber
from tqdm import tqdm
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, using OCR if necessary."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    
    if not text:
        images = convert_from_path(pdf_path)
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        for img in images:
            img_np = np.array(img)
            result = ocr.ocr(img_np, rec=True)
            for line in result[0]:
                text += line[1][0] + '\n'
    return text

def convert_pdfs_to_text(pdf_folder, output_folder):
    """Convert all PDF files in a folder to text files in an output folder."""
    print(f"Converting PDFs from {pdf_folder} to text files in {output_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(pdf_folder)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)

# finance folder
finance_pdf_folder = "../競賽資料集/reference/finance"  
finance_output_folder = "data/test"
convert_pdfs_to_text(finance_pdf_folder, finance_output_folder)
# insurance folder
# insurance_pdf_folder = "../競賽資料集/reference/insurance"
# insurance_output_folder = "data/insurance"
# convert_pdfs_to_text(insurance_pdf_folder, insurance_output_folder)