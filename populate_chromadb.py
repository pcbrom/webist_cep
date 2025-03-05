import polars as pl
import chromadb
import multiprocessing as mp
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import shutil
import os

# Configurations
N_THREADS = os.cpu_count() if os.cpu_count() else 8  # Use all available cores
BATCH_SIZE = 8192  # Reduction for lower memory usage
DB_PATH = "chroma_db"  # ChromaDB directory

# Remove previous database (if it exists)
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("ChromaDB database removed successfully")

# Create new database
print("Creating new ChromaDB database")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="ncm-all-data")

# Load data from CSV
with tqdm(total=100, desc="Loading data") as pbar:
    df = pl.read_csv('lobradouro', separator='\t', encoding='utf-8', n_threads=N_THREADS)
    pbar.update(100)

# Optimized processing: Create a single embedding per document
def format_document(row):
    text = f"NCM: {row['NCM']} | Rótulo: {row['rotulo']} | Produto: {row['XPROD']}"
    return {
        "text": text,  # Text to be transformed into embedding
        "metadata": {   # Metadata for advanced filtering
            "NCM": row['NCM'],
            "rotulo": row['rotulo'],
            "Item": row['Item'],
            "Produto": row['XPROD']
        }
    }

# Parallel processing of documents
with mp.Pool(processes=N_THREADS) as pool:
    documents = list(tqdm(pool.imap(format_document, df.to_dicts()), total=len(df), desc="Formatting data"))

# Generate embeddings for the formatted documents
texts = [doc["text"] for doc in documents]
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(texts, batch_size=1024, convert_to_tensor=True, show_progress_bar=True).tolist()

# Insertion into ChromaDB in batches
print("Populating ChromaDB")
with tqdm(total=len(documents), desc="Adding to ChromaDB") as pbar:
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]
        metadatas = [doc["metadata"] for doc in batch_docs]

        collection.add(
            embeddings=batch_embeddings,
            metadatas=metadatas,
            ids=[f"doc_{j}" for j in range(i, i + len(batch_docs))],
            documents=[doc["text"] for doc in batch_docs]
        )

        pbar.update(len(batch_docs))

print("ChromaDB populated successfully!")