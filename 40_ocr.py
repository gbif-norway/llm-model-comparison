import os
import pandas as pd
from google.cloud import vision
import io

# Set the environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/itf-fi-ml/home/michato/src/llm-model-comparison/secret/gapi-secret.json'

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

def perform_ocr(image_path):
    # Read the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create an Image object
    image = vision.Image(content=content)

    # Perform OCR on the image
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Extract the full text from the response
    if texts:
        return texts[0].description
    else:
        return ""

# Read the CSV file with scaled image paths
df = pd.read_csv('data/data_subset_with_scaled_paths.csv')

# Add a new column for OCR results
df['ocr_text'] = ''

# Process each scaled image
for index, row in df.iterrows():
    scaled_image_path = row['scaled_local_path']

    print(f"Processing OCR for: {scaled_image_path}")
    ocr_text = perform_ocr(scaled_image_path)

    # Update the DataFrame with the OCR text
    df.at[index, 'ocr_text'] = ocr_text
    print(f"OCR completed for: {scaled_image_path}")

# Save the updated DataFrame
df.to_csv('data/data_subset_with_ocr.csv', index=False)

print("OCR processing completed for all images")
print("Updated CSV saved as data/data_subset_with_ocr.csv")
