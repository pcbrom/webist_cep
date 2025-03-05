import os
from dotenv import load_dotenv
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import Dataset
# huggingface-cli download nicholasKluge/TeenyTinyLlama-160m --local-dir /mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/hf_models/vinidiol/TeenyTinyLlama --local-dir-use-symlinks False

# Specify the path to your .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"

# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# Define the local model path
model_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/hf_models/vinidiol/TeenyTinyLlama-160m-CEP-ft"
model_name = "TeenyTinyLlama-160m-CEP-ft"

# Check if the model directory exists
if not os.path.exists(model_path):
    raise OSError(f"Model directory does not exist: {model_path}. Download the model first.")

# Load the CSV file
csv_file = "augmented_prompt.csv"
df = pd.read_csv(csv_file, decimal='.', sep=',', encoding='utf-8')
print(df)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Load the local model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define the text generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)

def generate_response(example):
    augmented_prompt = example['augmented_prompt']
    temperature = 0.7
    top_p = 0.9

    response = llm_pipeline(
        augmented_prompt,
        do_sample=True,  # If True parameters such as temperature and top_p influence the generation of the text.
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=300
    )[0]['generated_text']
    response = response.replace(augmented_prompt, "")
    return {'results': response}

# Process the dataset
processed_dataset = dataset.map(generate_response)

# Update the DataFrame with the results
df['results'] = processed_dataset['results']

# Save the updated DataFrame
output_filename = f"results/experimental_design_results_{model_name}.csv"
df.to_csv(output_filename, index=False)

print(f"Completed. Results saved in {output_filename}")
