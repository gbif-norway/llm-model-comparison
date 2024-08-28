import pandas as pd
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import argparse
import os
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Ensure the API key is set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

# Set up argument parser
parser = argparse.ArgumentParser(description="Run OCR text through multiple language models")
parser.add_argument("--system_prompt", type=str, default="""
You are an adept herbarium digitization system, working on OCR text extracted from the images of scanned herbarium specimens.
First you correct any obvious OCR errors, and then you extract ONLY the following Darwin Core terms:

- scientificName: Full scientific name, not containing identification qualifications.
- catalogNumber: Unique identifier for the record in the dataset or collection.
- recordNumber: Identifier given during recording, often linking field notes and Occurrence record.
- recordedBy: List of people, groups, or organizations responsible for recording the original Occurrence.
- year: Four-digit year of the Event.
- month: Integer for the month of the Event.
- day: Integer for the day of the Event, not populated unless month and year are filled in.
- dateIdentified: Date when the subject was determined to represent the Taxon.
- identifiedBy: Person, group, or organization assigning the Taxon to the subject.
- verbatimIdentification: Taxonomic identification as it appeared in the original record.
- country: Name of the country or major administrative unit for the Location.
- countryCode: Standard code for the country of the Location.
- decimalLatitude: Geographic latitude in decimal degrees of the Location's center.
- decimalLongitude: Geographic longitude in decimal degrees of the Location's center.
- location: A spatial region or named place.
- minimumElevationInMeters: The lower limit of the range of elevation in meters.
- maximumElevationInMeters: The upper limit of the range of elevation in meters.
- verbatimElevation: The original description of the elevation.

If there are multiple valid values for a term, I separate them with "|". If I can't identify information for a specific term, and/or the term is blank, I skip the term in my response. I respond in minified JSON.
!IMPORTANT: I only respond with JSON and nothing else.
!IMPORTANT: You only use the data from the user prompt and do not create fictional data.
""")
parser.add_argument("--user_prompt", type=str, default='{ocr_text}')
parser.add_argument("--num_rows", type=int, default=10, help="Number of rows to process (default: 10, use -1 for all rows)")
args = parser.parse_args()

# Load the OCR results
df = pd.read_csv('data/data_subset_with_ocr.csv')

# Check if the results file already exists
results_file = 'data/data_subset_with_model_responses.csv'
if os.path.exists(results_file):
    existing_results = pd.read_csv(results_file)
    df = pd.merge(df, existing_results, on='occurrenceID', how='left', suffixes=('', '_existing'))
else:
    existing_results = None

# Define the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(args.system_prompt),
    HumanMessagePromptTemplate.from_template(args.user_prompt)
])

# Set up different models
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# Enable multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set pad_token_id to eos_token_id
tokenizer.pad_token_id = model.module.config.eos_token_id if hasattr(model, 'module') else model.config.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model.module if hasattr(model, 'module') else model,  # Use model.module if it's DataParallel
    tokenizer=tokenizer,
    max_new_tokens=1024,
    device=device,  # Use the determined device
)

models = {
    "llama-3.1-8b": HuggingFacePipeline(pipeline=pipe),
    "gpt-4o": ChatOpenAI(model_name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
    "gpt-4": ChatOpenAI(model_name="gpt-4", api_key=os.getenv("OPENAI_API_KEY")),
    "gpt-4o-mini": ChatOpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    "gpt-4-turbo": ChatOpenAI(model_name="gpt-4-turbo", api_key=os.getenv("OPENAI_API_KEY")),

    # Add more models as needed
}

# Function to run prompt against a model
def run_prompt(model, ocr_text):
    chain = chat_prompt | model
    response = chain.invoke({"ocr_text": ocr_text})

    # Handle different response formats
    if isinstance(response, str):
        # For HuggingFacePipeline (Llama model)
        # Find the last JSON object in the response
        import re
        json_matches = re.findall(r'\{.*?\}', response, re.DOTALL)
        if json_matches:
            return json_matches[-1].strip()
        else:
            return response.strip()
    elif hasattr(response, 'content'):
        # For ChatOpenAI models
        return response.content.strip()
    else:
        raise ValueError(f"Unexpected response format: {type(response)}")

# Function to run prompt against a model and save results
def run_prompt_and_save(row, model, model_name, results_file):
    if pd.isnull(row.get(f'{model_name}_response')):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = run_prompt(model, row['ocr_text'])
                if response:
                    row[f'{model_name}_response'] = response
                    # Save the updated row to the CSV file
                    row.to_frame().T.to_csv(results_file, mode='a', header=False, index=False)
                    return response
                elif attempt == max_retries - 1:
                    # If this is the last attempt and still empty, store 'FAIL'
                    row[f'{model_name}_response'] = 'FAIL'
                    row.to_frame().T.to_csv(results_file, mode='a', header=False, index=False)
                    return 'FAIL'
                else:
                    print(f"Empty response from {model_name} for occurrenceID {row['occurrenceID']}. Retrying...")
            except Exception as e:
                error_message = f"Error processing {model_name} for occurrenceID {row['occurrenceID']} (Attempt {attempt+1}/{max_retries}): {str(e)}"
                print(error_message)
                # Log the error to a file
                with open('error_log.txt', 'a') as f:
                    f.write(f"{error_message}\n")
                if attempt == max_retries - 1:
                    # If this is the last attempt, store 'FAIL'
                    row[f'{model_name}_response'] = 'FAIL'
                    row.to_frame().T.to_csv(results_file, mode='a', header=False, index=False)
                    return 'FAIL'
    return row[f'{model_name}_response']

# Process OCR text with different models
for model_name, model in models.items():
    print(f"Processing {model_name} for images without existing results")

    # Check if the column exists and process only rows without results
    if f'{model_name}_response' not in df.columns or df[f'{model_name}_response'].isnull().any():
        # Use tqdm to create a progress bar
        tqdm.pandas(desc=f"Processing {model_name}")

        # Determine the number of rows to process
        num_rows = len(df) if args.num_rows == -1 else min(args.num_rows, len(df))

        try:
            # Process the specified number of rows
            df[f'{model_name}_response'] = df.head(num_rows).progress_apply(
                lambda row: run_prompt_and_save(row, model, model_name, results_file),
                axis=1
            )
        except Exception as e:
            print(f"Error occurred while processing {model_name}: {str(e)}")
            print("Stopping execution.")
            break

# Remove any columns with '_existing' suffix
df = df.loc[:, ~df.columns.str.endswith('_existing')]

# Save final results to CSV (this will overwrite the file with the complete dataset)
df.to_csv(results_file, index=False)

print(f"Model comparison completed. Results saved to {results_file}")
