import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle # For saving/loading BERTopic model and FAISS index
from umap import UMAP
from hdbscan import HDBSCAN

# Import the Google Generative AI library
import google.generativeai as genai

# -------- CONFIG --------
CSV_PATH = "Kisan Call center Queries.csv"
GOOGLE_API_KEY = "
" 

# Initialize the Google Generative AI client
if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY_HERE":
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("Google API Key is not set or is still the placeholder. Please replace 'YOUR_GOOGLE_API_KEY_HERE' with your actual key.")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash" # Or gemini-pro, gemini-1.5-pro
EMBEDDINGS_FILE = "document_embeddings.npy"
TOPIC_MODEL_FILE = "bertopic_model.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
TOPIC_IDS_FILE = "topic_ids.pkl"
TOPIC_DOCS_FILE = "topic_docs.pkl"

# -------- STEP 1: Load Data --------
df = pd.read_csv(CSV_PATH)
if 'questions' not in df.columns or 'answers' not in df.columns:
    raise ValueError("CSV must contain 'questions' and 'answers' columns")

df['questions'] = df['questions'].fillna('')
df['answers'] = df['answers'].fillna('')
df['combined'] = df['questions'] + " " + df['answers']

print(f"Loading embedding model: {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)

# -------- STEP 2: Pre-compute/Load Embeddings --------
document_embeddings = None
if os.path.exists(EMBEDDINGS_FILE):
    print(f"Loading pre-computed document embeddings from {EMBEDDINGS_FILE}...")
    loaded_embeddings = np.load(EMBEDDINGS_FILE)
    if len(loaded_embeddings) == len(df['combined']):
        document_embeddings = loaded_embeddings
    else:
        print("Warning: Number of loaded embeddings does not match documents. Recomputing document embeddings.")
        document_embeddings = None # Force re-computation if mismatch

if document_embeddings is None:
    print(f"Generating embeddings for {len(df['combined'])} documents...")
    document_embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True, batch_size=128)
    np.save(EMBEDDINGS_FILE, document_embeddings)
    print(f"Document embeddings saved to {EMBEDDINGS_FILE}")

# -------- STEP 3: Topic Modeling with BERTopic --------
topic_model = None
topics = None # Initialize topics outside the if block

if os.path.exists(TOPIC_MODEL_FILE):
    print(f"Loading pre-computed BERTopic model from {TOPIC_MODEL_FILE}...")
    with open(TOPIC_MODEL_FILE, 'rb') as f:
        topic_model = pickle.load(f)
    print("Re-calculating topics for the loaded BERTopic model (fast operation)...")
    topics, _ = topic_model.transform(df['combined'], embeddings=document_embeddings)
    df['topic'] = topics
else:
    print("🔍 Generating topics using BERTopic...")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42, low_memory=True)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics="auto",
        top_n_words=10,
        verbose=True,
        embedding_model=model
    )

    topics, _ = topic_model.fit_transform(df['combined'], embeddings=document_embeddings)
    df['topic'] = topics

    print(f"Saving BERTopic model to {TOPIC_MODEL_FILE}...")
    with open(TOPIC_MODEL_FILE, 'wb') as f:
        pickle.dump(topic_model, f)
    print("BERTopic model saved.")


# -------- STEP 4: Create Topic Documents and FAISS Index --------
topic_docs = {}
topic_ids = []
index = None # Initialize index outside if block

if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(TOPIC_IDS_FILE) and os.path.exists(TOPIC_DOCS_FILE):
    print(f"Loading pre-computed FAISS index and topic data...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(TOPIC_IDS_FILE, 'rb') as f:
        topic_ids = pickle.load(f)
    with open(TOPIC_DOCS_FILE, 'rb') as f:
        topic_docs = pickle.load(f)

    print("FAISS index and topic data loaded.")
else:
    print("📝 Creating topic documents...")
    topic_docs = {
        topic: " ".join(df[df['topic'] == topic]['combined'])
        for topic in df['topic'].unique()
    }
    topic_ids = list(topic_docs.keys())
    topic_texts = list(topic_docs.values())

    print("📦 Generating embeddings and building FAISS index for topics...")
    topic_embeddings = model.encode(topic_texts, show_progress_bar=True, batch_size=64)

    embedding_dim = topic_embeddings[0].shape[0]

    nlist = min(100, len(topic_embeddings) // 4)
    if nlist == 0: nlist = 1

    quantizer = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)

    print(f"Training FAISS IndexIVFFlat with nlist={nlist}...")
    if len(topic_embeddings) >= nlist * 39: # Heuristic for sufficient training data
        index.train(topic_embeddings)
    else:
        print("Warning: Not enough topic embeddings for optimal FAISS training. Using IndexFlatL2 instead.")
        index = faiss.IndexFlatL2(embedding_dim) # Fallback to flat if not enough data for IVF training

    index.add(np.array(topic_embeddings))

    if isinstance(index, faiss.IndexIVFFlat):
        index.nprobe = max(1, nlist // 10)

    print(f"Saving FAISS index to {FAISS_INDEX_FILE}...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"Saving topic IDs to {TOPIC_IDS_FILE}...")
    with open(TOPIC_IDS_FILE, 'wb') as f:
        pickle.dump(topic_ids, f)
    print(f"Saving topic documents to {TOPIC_DOCS_FILE}...")
    with open(TOPIC_DOCS_FILE, 'wb') as f:
        pickle.dump(topic_docs, f)
    print("FAISS index and topic data saved.")

# -------- STEP 5: Helper Functions --------
def find_relevant_topic(query):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    topic_id = topic_ids[I[0][0]]
    return topic_id

def generate_response(user_query):
    topic_id = find_relevant_topic(user_query)
    context = topic_docs[topic_id]

    prompt_parts = [
        {"role": "user", "parts": [f"""
Answer the following agricultural question accurately and concisely, using only the provided context. If the answer is not in the context, state that you don't know.

Context:
{context}

Question:
{user_query}

Answer:"""
        ]}
    ]

    try:
        gemini_model = genai.GenerativeModel(LLM_MODEL)
        response = gemini_model.generate_content(
            prompt_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=250
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response from Google Gemini API: {e}")
        return "I apologize, but I could not generate a response at this time."


# -------- STEP 6: Chat Loop --------
if __name__ == "__main__":
    print("🌾 Agriculture Chatbot Ready! Ask your question or type 'exit' to quit.")
    response_cache = {}

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ("exit", "quit"):
            print("🛑 Exiting. Thank you!")
            break

        if user_input in response_cache:
            print("🤖 Bot (cached):", response_cache[user_input])
            continue

        answer = generate_response(user_input)
        response_cache[user_input] = answer
        print("🤖 Bot:", answer)
