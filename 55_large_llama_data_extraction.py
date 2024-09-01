import pandas as pd
import os
import requests

from gradio_client import Client



# Update the API URL to match your Hugging Face Space endpoint
# API_URL = "https://rukaya.hf.space/api/predict"  # Replace with your actual Space URL
# headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_HUB_TOKEN')}"}

# Load the system prompt
with open('system_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read().strip()

# Function to query the Hugging Face Space
def query_hf_space(ocr_text):
    client = Client("rukaya/llm-comparison")
    result = client.predict(
            system_prompt=SYSTEM_PROMPT,
            ocr_text=ocr_text,
            api_name="/predict"
    )
    print(result)
    return result
    # payload = {'data': [f'{SYSTEM_PROMPT}\n\nOCR TEXT:\n{ocr_text}']}
    # response = requests.post(API_URL, headers=headers, json=payload)
    # if response.status_code == 200:
    #     # Adjust the processing of the response as needed
    #     return response.json()
    # else:
    #     print(f"Error: {response.status_code}, {response.text}")
    #     return None

# Load OCR text data
df = pd.read_csv('data/data_subset_with_ocr.csv', dtype='str', usecols=['id', 'ocr_text'])
df = df.head(1)  # Truncate the dataframe to the first 10 rows

results = []
for _, row in df.iterrows():
    ocr_text = row['ocr_text']
    # Query the Space with the combined prompt
    result = query_hf_space(ocr_text)
    results.append(result)
print(results)
# Add the results as a new column in the dataframe
df['json_results'] = results
df.to_csv('data/data_subset_with_llama405_response.csv', index=False)
