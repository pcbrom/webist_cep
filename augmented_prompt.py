import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import os
import json

# Load API Key from .env file
dotenv_path = "/mnt/4d4f90e5-f220-481e-8701-f0a546491c35/arquivos/projetos/.env"
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Error: OPENAI_API_KEY not found. Check your .env file.")

openai.api_key = openai_api_key
model = 'gpt-4o-mini-2024-07-18'

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="ncm-all-data")

# Load Sentence Transformer model
model_sentence = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Load DataFrame
plan_file_path = "experimental_design_plan_prompt_baseline.xlsx"
if not os.path.exists(plan_file_path):
    raise FileNotFoundError(f"Error: File {plan_file_path} not found.")

df = pd.read_excel(plan_file_path)

# Function to Generate Augmented Prompt Using RAG
def create_augmented_prompt(prompt):
    """Generates an augmented prompt using RAG (Retrieval-Augmented Generation)."""
    filters = None
    try:
        # Extract metadata using OpenAI
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"Tente identificar e extrair os possíveis metadados NCM, rótulo, Item e Produto da seguinte pergunta. Retorne um JSON. Se não encontrar a informação retorne null. Importante rótulo é igual ao produto, se encontrar um deles preenher automaticamente o outro com o conteúdo. Pergunta: {prompt}"}
            ],
            temperature=0.0
        )

        metadata_str = response.choices[0].message.content.strip()
        if metadata_str.startswith("```json") and metadata_str.endswith("```"):
            metadata_str = metadata_str.strip("```").strip()
        if metadata_str.startswith("json"):
            metadata_str = metadata_str[4:].lstrip()
        metadata = json.loads(metadata_str)

        filters = {}
        if isinstance(metadata, dict):
            filters = {k: v for k, v in metadata.items() if v is not None}


    except (openai.OpenAIError, json.JSONDecodeError) as e:
        print(f"Error extracting metadata: {e}")
    
    # Retrieve context from ChromaDB
    prompt_embeddings = model_sentence.encode([prompt], convert_to_tensor=True).tolist()
    results = None
    results_no_filters = collection.query(
        query_embeddings=prompt_embeddings,
        n_results=5,
        include=["documents", "embeddings", "distances"]
    )

    if "documents" not in results_no_filters or not results_no_filters["documents"]:
        context = "No relevant information found.\n"
    else:
        context = "\n".join([doc.strip() for doc in results_no_filters["documents"][0]]) + "\n"

    if filters:
        # Handle multiple filters by iterating through them
        for key, value in filters.items():
            temp_results = collection.query(
                query_embeddings=prompt_embeddings,
                n_results=5,
                where={key: value},  # Apply filter for each key-value pair
                include=["documents", "embeddings", "distances"]
            )
            if results is None:
                results = temp_results
            else:
                # Combine results (you might need a more sophisticated combination logic)
                if "documents" in temp_results and temp_results["documents"]:
                    results["documents"][0].extend(temp_results["documents"][0])

    if results and "documents" in results and results["documents"]:
        context += "\n".join([doc.strip() for doc in results["documents"][0]]) + "\n"
    else:
        context += "No relevant information found.\n"

    augmented_prompt = f"""
    Você é um assistente especializado em responder de forma objetiva e clara às perguntas com base em informações relevantes extraídas de uma base de conhecimento. 

    Contexto relevante recuperado:

    {context}

    Pergunta:
    {prompt}

    Se o contexto não contiver informações suficientes, indique que não há dados suficientes para responder com segurança.
    """
    return augmented_prompt

tqdm.pandas(desc="Generating augmented prompts")

# Create a temporary series with augmented prompts
tmp_unique = {}
for prompt in tqdm(df['prompt'].unique(), desc="Generating augmented prompts"):
    tmp_unique[prompt] = create_augmented_prompt(prompt)
df['augmented_prompt'] = df['prompt'].map(tmp_unique)

# Save Results
df.to_csv("augmented_prompt.csv", index=False)
print("Results saved in augmented_prompt.csv")