# 🌾 Agriculture Chatbot

An intelligent chatbot designed to answer agricultural queries. This project integrates a Flask web application with advanced NLP techniques such as **BERTopic** for topic modeling, **FAISS** for similarity search, and **Google's Gemini API** for generating context-aware responses.

## ✨ Features

- **🔍 Intelligent Query Routing**  
  Uses Sentence Transformers and FAISS to retrieve the most relevant agricultural topic from the dataset.
- **🧠 Context-Aware AI Responses**  
  Utilizes `gemini-1.5-flash` to provide accurate answers strictly grounded in the retrieved topic.
- **📚 Automated Topic Discovery**  
  Employs BERTopic to uncover hidden themes within the dataset and structure the knowledge base.
- **⚡ High-Performance Retrieval**  
  FAISS enables fast and efficient semantic search over embeddings.
- **💬 Interactive Web Interface**  
  Built with Flask, Bootstrap, HTML/CSS, and JavaScript for an intuitive chatbot experience.
- **🚀 Optimized Performance**  
  Caches embeddings, topic models, and indexes for faster startup and response time.
