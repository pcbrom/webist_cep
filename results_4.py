import os
from dotenv import load_dotenv
import pandas as pd
import openai
from tqdm import tqdm
import time

# Specify the path to your .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"

# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# Access and store the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
model = 'gpt-4o-mini-2024-07-18'

# Import model
csv_file = "augmented_prompt.csv"
df = pd.read_csv(csv_file, decimal='.', sep=',', encoding='utf-8')
df = df[df['model'] == model]
cols_to_fill = ['results', 'score']
df[cols_to_fill] = df[cols_to_fill].fillna('')
print(df)

# Iterate over each row and make API call
output_filename = f"experimental_design_results_{model}.csv"
for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {model}"):
    if index % 100 == 0 and index != 0:
        print("min. pause...")
        time.sleep(60)
    try:
        augmented_prompt = row['augmented_prompt']

        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=float(row['temperature']),
            top_p=float(row['top_p'])
        )

        # Extract and store the generated text
        generated_text = response.choices[0].message.content
        df.loc[index, 'results'] = generated_text

    except openai.OpenAIError as e:
        print(f"Error processing row {index} for model {model}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"  # Store the error message
    except Exception as e:
        print(f"Unexpected error processing row {index} for model {model}: {e}")
        df.loc[index, 'results'] = f"Error: {e}"

# Save the updated DataFrame (optional)
df.to_csv(output_filename, index=False)