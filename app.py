from flask import Flask, render_template, request, jsonify
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from umap import UMAP
from hdbscan import HDBSCAN
import google.generativeai as genai

app = Flask(__name__)

# -------- CONFIG (Same as before) --------
CSV_PATH = "Kisan Call center Queries.csv"
GOOGLE_API_KEY = "copy and paste here" 

# Initialize the Google Generative AI client
if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY_HERE":
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("Google API Key is not set or is still the placeholder. Please replace 'YOUR_GOOGLE_API_KEY_HERE' with your actual key obtained from Google AI Studio.")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-1.5-flash"
EMBEDDINGS_FILE = "document_embeddings.npy"
TOPIC_MODEL_FILE = "bertopic_model.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
TOPIC_IDS_FILE = "topic_ids.pkl"
TOPIC_DOCS_FILE = "topic_docs.pkl"

# Global variables to store loaded models/data
df = None
model = None
topic_model = None
topic_docs = {}
topic_ids = []
index = None
response_cache = {} # Cache for chatbot responses

def load_all_models_and_data():
    """Loads all necessary data and models into global variables."""
    global df, model, topic_model, topic_docs, topic_ids, index

    print("--- Starting Model and Data Loading ---")

    # Load Data
    df = pd.read_csv(CSV_PATH)
    if 'questions' not in df.columns or 'answers' not in df.columns:
        raise ValueError("CSV must contain 'questions' and 'answers' columns")
    df['questions'] = df['questions'].fillna('')
    df['answers'] = df['answers'].fillna('')
    df['combined'] = df['questions'] + " " + df['answers']
    print("Data loaded.")

    # Load Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")

    # Load/Generate Embeddings
    document_embeddings = None
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Loading pre-computed document embeddings from {EMBEDDINGS_FILE}...")
        loaded_embeddings = np.load(EMBEDDINGS_FILE)
        if len(loaded_embeddings) == len(df['combined']):
            document_embeddings = loaded_embeddings
        else:
            print("Warning: Number of loaded embeddings does not match documents. Recomputing document embeddings.")
            document_embeddings = None
    if document_embeddings is None:
        print(f"Generating embeddings for {len(df['combined'])} documents...")
        document_embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True, batch_size=128)
        np.save(EMBEDDINGS_FILE, document_embeddings)
        print(f"Document embeddings saved to {EMBEDDINGS_FILE}")
    print("Document embeddings ready.")

    # Load/Train BERTopic Model
    topics = None
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
    print("Topic model ready.")

    # Create/Load Topic Documents and FAISS Index
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
    print("--- Model and Data Loading Complete ---")


def find_relevant_topic(query):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    topic_id = topic_ids[I[0][0]]
    return topic_id

def generate_llm_response(user_query, context):
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


@app.route('/')
def index():
    """Renders the main chatbot HTML page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handles chatbot queries from the frontend."""
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"answer": "Please provide a query."}), 400

    # Check cache first
    if user_query in response_cache:
        print("Returning cached response.")
        return jsonify({"answer": response_cache[user_query]})

    # Find relevant topic and generate response
    topic_id = find_relevant_topic(user_query)
    context = topic_docs[topic_id]
    answer = generate_llm_response(user_query, context)

    # Cache the response
    response_cache[user_query] = answer

    return jsonify({"answer": answer})

if __name__ == '__main__':
    # Load models and data only once when the Flask app starts
    load_all_models_and_data()
    print("Flask app starting...")
    app.run(debug=True) # debug=True is good for development, disable in production