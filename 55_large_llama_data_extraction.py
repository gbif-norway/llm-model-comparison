import pandas as pd
import os
import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-405B"
headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_HUB_TOKEN')}"}

with open('system_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read().strip()

def query_hf_api(ocr_text):
    payload = { 'inputs': f'{SYSTEM_PROMPT}\n\nOCR TEXT:\n{ocr_text}' }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

df = pd.read_csv('data/data_subset_with_ocr.csv', dtype='str', usecols=['id', 'ocr_text'])
results = []
for _, row in df.iterrows():
    ocr_text = row['ocr_text']
    result = query_hf_api(ocr_text)
    results.append(result)

df['json_results'] = results
import pdb; pdb.set_trace()
df.to_csv('data/data_subset_with_llama405_response.csv', index=False)

