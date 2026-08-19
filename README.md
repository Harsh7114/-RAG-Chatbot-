📄 RAG Chatbot — Agentic PDF Question Answering

An Agentic RAG (Retrieval-Augmented Generation) chatbot built with LangChain, LangGraph, Groq, Hugging Face embeddings, and Streamlit.

The application allows users to upload one or more PDF documents and ask questions about their contents. The agent uses a retrieval tool to find relevant information from the uploaded documents before generating an answer.

🚀 Live Demo

👉 https://jxrxogjvyfgfevfr2dgbpu.streamlit.app/

✨ Features

📤 Upload one or multiple PDF documents

📚 Extract and split PDF content into smaller chunks

🔎 Semantic search using vector embeddings

🤖 Agentic RAG using a LangChain agent

🛠️ Custom retrieval tool for fetching relevant document context

💬 Interactive Streamlit chat interface

🧠 Conversation state using Streamlit session state

⚡ Groq-powered LLM inference

🔐 API key stored securely using Streamlit Secrets

🏗️ Architecture

PDF Upload
    ↓
PyPDFDirectoryLoader
    ↓
RecursiveCharacterTextSplitter
    ↓
Hugging Face Embeddings
    ↓
InMemoryVectorStore
    ↓
retrieve_context Tool
    ↓
LangChain Agent + Groq LLM
    ↓
Answer
    ↓
Streamlit Chat UI

🧠 How Agentic RAG Works

The user uploads PDF document(s).

The PDF text is extracted using PyPDFDirectoryLoader.

The document is divided into chunks using RecursiveCharacterTextSplitter.

all-MiniLM-L6-v2 converts the chunks into vector embeddings.

The embeddings are stored in an in-memory vector store.

The LangChain agent receives the user's question.

When external document knowledge is required, the agent calls the retrieve_context tool.

The tool performs similarity search and returns the most relevant chunks.

The LLM uses the retrieved context to generate the final answer.

🛠️ Tech Stack

Python

Streamlit

LangChain

LangGraph

Groq

Hugging Face Sentence Transformers

PyPDF

InMemoryVectorStore

📁 Project Structure

RAG-Chatbot/
│
├── apps/
│   ├── rag_agent_app2.py
│   └── requirements.txt
│
├── .gitignore
└── README.md

⚙️ Run Locally

1. Clone the repository

git clone https://github.com/Harsh7114/-RAG-Chatbot-.git
cd -RAG-Chatbot-

2. Create and activate a virtual environment

python -m venv env

Windows:

env\Scripts\activate

3. Install dependencies

pip install -r apps/requirements.txt

4. Configure the Groq API key

For local development, create a .env file:

GROQ_API_KEY="your_groq_api_key"

For Streamlit Cloud, add the same key under:

App → Settings → Secrets

GROQ_API_KEY = "your_groq_api_key"

⚠️ Never commit .env or your API key to GitHub.

5. Run the application

streamlit run apps/rag_agent_app2.py

☁️ Deployment

This project is deployed using Streamlit Community Cloud.

Deployment configuration:

Repository: Harsh7114/-RAG-Chatbot-

Branch: main

Main file: apps/rag_agent_app2.py

Secret: GROQ_API_KEY

🔐 Security

API keys are not stored in the source code. The deployed application reads the Groq API key from Streamlit Secrets.

Make sure files such as .env, virtual environments, and other local/private files are included in .gitignore.

⚠️ Current Limitations

The vector store is stored in memory, so uploaded documents are not permanently persisted.

Documents need to be uploaded again after a new application session.

Retrieval quality depends on chunking and embedding quality.

The application is designed for question answering over uploaded documents rather than general-purpose web search.

📌 Future Improvements

Add persistent vector databases such as Chroma, FAISS, or Pinecone

Add document management and deletion

Add source/page citations in answers

Improve conversation memory

Add streaming responses

Add support for more document formats

Add authentication and user-specific document collections

👨‍💻 Author

Harsh Ranjan

Built as a learning project to understand RAG, Agentic RAG, LangChain Agents, tool calling, vector embeddings, and LLM-powered applications.

⭐ If you find this project useful, consider giving the repository a star!
