import pandas as pd
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Run OCR text through multiple language models")
parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant that analyzes OCR text.")
parser.add_argument("--user_prompt", type=str, default="Analyze the following OCR text and provide a brief summary: {ocr_text}")
args = parser.parse_args()

# Load the OCR results
df = pd.read_csv('data/data_subset_with_ocr.csv')

# Define the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(args.system_prompt),
    HumanMessagePromptTemplate.from_template(args.user_prompt)
])

# Set up different models
models = {
    "gpt-4": OpenAI(model_name="gpt-4"),
    "gpt-3.5-turbo": OpenAI(model_name="gpt-3.5-turbo"),
    "llama-3.1-8b": HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        task="text-generation",
        model_kwargs={"temperature": 0.7, "max_length": 256}
    ),
    # Add more models as needed
}

# Function to run prompt against a model
def run_prompt(model, ocr_text):
    chain = LLMChain(llm=model, prompt=chat_prompt)
    return chain.run(ocr_text=ocr_text)

# Process OCR text with different models
for model_name, model in models.items():
    print(f"Processing {model_name} for all images")
    df[f'{model_name}_response'] = df['ocr_text'].apply(lambda x: run_prompt(model, x))

# Save results to CSV
df.to_csv('data/data_subset_with_model_responses.csv', index=False)

print("Model comparison completed. Results saved to data/data_subset_with_model_responses.csv")
