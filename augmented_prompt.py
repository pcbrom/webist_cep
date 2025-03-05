import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import chromadb

# File path
file_path = "Amostra100AvalCEP.txt"

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="cep")

# Load Sentence Transformer model
model_sentence = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Read file content
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Process content to extract prompts and baselines
data = []
for i in range(0, len(lines)):
    prompt_line = re.search(r'"content": responda a seguinte pergunta: (.*?)",', lines[i])
    if prompt_line:
        prompt = prompt_line.group(1)
    else:
        prompt = None

    baseline_line = re.search(r'"content":\s"(.*?)",\s"role":\s"assistant"', lines[i])
    if baseline_line:
        baseline = baseline_line.group(1)
    else:
        baseline = None

    if prompt and baseline: # only append if both prompt and baseline are not None
        data.append({"prompt": prompt, "baseline": baseline})
 
# Create DataFrame
df = pd.DataFrame(data)

def retrieve_context(prompt, model_sentence, collection):
    """Retrieves context from ChromaDB based on the given prompt."""
    prompt_embeddings = model_sentence.encode([prompt], convert_to_tensor=True).tolist()
    results = collection.query(
        query_embeddings=prompt_embeddings,
        n_results=5,
        include=["documents"]
    )

    if "documents" not in results or not results["documents"]:
        context = "No relevant information found.\n"
    else:
        context = "\n".join([doc.strip() for doc in results["documents"][0]]) + "\n"
    return context

df['context'] = df['prompt'].apply(lambda x: retrieve_context(x, model_sentence, collection))

df['augmented_prompt'] = df.apply(lambda row: f"""Você é um assistente especializado em responder de forma objetiva e clara às perguntas com base em informações relevantes extraídas de uma base de conhecimento. 

Contexto relevante recuperado:

{row['context']}

Pergunta:
{row['prompt']}

Se o contexto não contiver informações suficientes, indique que não há dados suficientes para responder com segurança.""", axis=1)

print(df[['prompt', 'context', 'augmented_prompt']].head())

df.to_csv('augmented_prompt.csv', index=False)
