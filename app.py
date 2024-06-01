import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import pandas as pd
import torch
from langchain_groq.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

images_directory = "data"

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
reader = easyocr.Reader(['en'], gpu=device.type == 'cuda')

# Image Preprocessing and extracting text using Tesseract
def preprocess_and_extract_text_tesseract(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    blurred_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)

    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = gray_image[y:y+h, x:x+w]

    pil_image = Image.fromarray(cropped_image)

    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(2)

    sharpened_image = enhanced_image.filter(ImageFilter.SHARPEN)

    sharpened_image_np = np.array(sharpened_image)

    kernel = np.ones((2, 2), np.uint8)
    morphed_image = cv2.morphologyEx(sharpened_image_np, cv2.MORPH_CLOSE, kernel)

    custom_config = r'--oem 3 --psm 6'
    tesseract_text = pytesseract.image_to_string(morphed_image, config=custom_config)

    return tesseract_text

# extracting text using EasyOCR
def extract_text_easyocr(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image {image_path}")
        return ""

    if image.shape[0] == 0 or image.shape[1] == 0:
        print(f"Error: Image {image_path} has invalid dimensions")
        return ""

    try:
        result = reader.readtext(image_path, detail=0)
        easyocr_text = "\n".join(result)
        return easyocr_text
    except Exception as e:
        print(f"Error processing image {image_path} with EasyOCR: {e}")
        return ""

# Features Extraction using LLM(Llama-3)
def extract_important_info(easyocr_text, tesseract_text):
    llm = ChatGroq(
        groq_api_key="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", #Note: I will be deleting this key after 9th june 2024
        model_name="Llama3-70b-8192"
    )

    template = PromptTemplate(
        input_variables=["tesseract_text", "easyocr_text"],
        template="""
        I have extracted text from same image using 2 methods i.e. Tesseract and EasyOCR. I am providing both extracted text here.
        Here is the extracted information from Tesseract:
        {tesseract_text}
        And here is the extracted information from EasyOCR:
        {easyocr_text}

        You can use both texts (tesseract_text, easyocr_text) and extract important information. Note: Both texts are extracted from same image.
        
        -----------------------------------------------------------

        Most Importat: follow mentioned format below strictly. i.e. every character should match same format mentioned below in your response(dont include any * before or after it)
        **don't include** any introductory statement like 'Here is the extracted information' etc.
        All below attributes are case sensetive, that's why I have named all in small case, while providing response you also follow same format.
        registration number:
        registration date:
        manufacturing date:
        chassis number:
        engine number:
        name:
        -------------------------------------------------------------

        Here is some additional information about every features format:

        registration number:
        note: if you find any registration number then extrat it in following format.
        Common labels: REGN NO, NO, Reg NO, Reg. Num, Na
        contains only uppercase letters, numbers, and special characters like hyphens (-) or spaces.
        Examples: HR1O-P-0840, DL2CAU7997, HR06S 8814
        
        registration date:
        Common labels: REG DT, DT, Reg DT
        Format: DD/MM/YYYY
        Example: 14/02/2013
        if date is in 14022013 format then bring it to format my placing / in between them. e.g: 18082014 becomes 18/08/2014
        Correct misdetected characters to /
        
        manufacturing date:
        note: if you find any manufaturing date then extrat it in following format.
        Common labels: MFG DT, Month/Year of mfg, mg DT, MFG DT
        Format: MM/YYYY
        Example: 12/2012
        if date is in 1212012 format then bring it to format my placing / in between them. e.g: 0312005 becomes 03/2005
        Correct misdetected characters to /
        
        chassis number:
        note: if you find any chassis number then extract it in following format.
        Common labels: CH NO, CH, CHINO
        Uppercase letters and numbers only
        Examples: M3EUA61S00170565, 607146KRZWOO622
        Convert to uppercase if mixed case
        
        engine number:
        note: if you find any engine number then extrat it in following format.
        Common labels: E NO, Engine, ENO, Enfine, E NO:
        Uppercase letters and numbers only
        Examples: FSDN4958005, E NO: G4EB9M256677
        Convert to uppercase if mixed case
        
        name:
        note: if you find name then extrat it in following format.
        Common labels: name, owner name
        No apostrophes or special characters
        Examples: MANJEET SINGH, Owner Name: John DSouza, Name: Raj Kumar Gupta

        
        If any information is not found, return "Not Found" for that field.
        """
    )

    formatted_prompt = template.format(easyocr_text=easyocr_text, tesseract_text=tesseract_text)

    messages = [HumanMessage(content=formatted_prompt)]

    response = llm.invoke(messages)

    extracted_info = response.content
    print("Extracted Info:", extracted_info)

    # Parsing extracted information
    info_lines = extracted_info.split('\n')
    info_dict = {}

    for line in info_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            info_dict[key.strip()] = value.strip()

    return info_dict

# Creating CSV file
def create_csv_with_text_and_image_names(directory, csv_filename):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)

            tesseract_text = preprocess_and_extract_text_tesseract(image_path)

            easyocr_text = extract_text_easyocr(image_path)

            extracted_info = extract_important_info(easyocr_text, tesseract_text)

            file_data = {
                'file_name': filename,
                'tesseract_text': tesseract_text,
                'easyocr_text': easyocr_text
            }

            for key, value in extracted_info.items():
                if value != 'Not Found':
                    file_data[key] = value

            data.append(file_data)

    df = pd.DataFrame(data)

    desired_columns = ['file_name', 'tesseract_text', 'easyocr_text', 'registration number', 'registration date', 'manufacturing date', 'chassis number', 'engine number',  'name']
    for col in desired_columns:
        if col not in df.columns:
            df[col] = 'Not Found'

    df = df.loc[:, desired_columns]

    print(df)

    df.to_csv(csv_filename, index=False)

create_csv_with_text_and_image_names(images_directory, 'output.csv')

#Note: To display output from output.csv to web interface 'streamlit run Show.py' after execution of this file