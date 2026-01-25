# 🎥 AskTheYouTube: Video Q&A RAG System

**AskTheYouTube** is a Retrieval-Augmented Generation (RAG) system that allows users to "chat" with YouTube videos. It leverages advanced embedding techniques and vector databases to provide context-aware answers to user queries based on video transcripts.

---

## 🚀 Features
- **Transcript Processing:** Automated fetching and chunking of YouTube video content.
- **Semantic Search:** Uses vector embeddings to find the most relevant video segments.
- **AI-Powered Answers:** Integrates with LLMs to generate precise responses based on retrieved context.
- **Enterprise Security:** Implements **Google Cloud Secret Manager** for secure API key handling.

---

## 🛠 Tech Stack
- **Language:** Python
- **Vector Database:** Pinecone
- **AI Models:** Google Vertex AI / OpenAI
- **Infrastructure:** Docker, Google Cloud Platform (GCP)
- **Frontend:** HTML/JS (Firebase Hosting ready)

---

## 🏗 Project Structure

- **`/services`**: Core RAG logic (Chunking, Embedding, Retrieval).
- **`/utils`**: Security utilities and API clients.
- **`/frontend`**: User interface for interacting with the system.
- **`main.py`**: Primary entry point for the backend service.

---

## 🔧 Setup & Installation

### 1. Clone the Repository
bash
git clone https://github.com/sandeepjanapati/asktheyoutube.git
cd asktheyoutube


### 2. Environment Configuration
Create a `.venv` and install dependencies:
bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt


### 3. Security Setup
This project uses **Google Cloud Secret Manager**. Ensure you have the following secrets configured in your GCP project:
- `pinecone-api-key`
- `vertex-ai-key` (or equivalent)

For local development, you can use a `.env` file (ensure it is ignored by Git).

---

## 📦 Deployment
The project includes a `Dockerfile` for containerized deployment to **Cloud Run** or **GKE**.

bash
docker build -t asktheyoutube-api .