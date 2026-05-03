# RAG-based AI Q&A Assistant with Google Gemini

This project is a CLI-based Retrieval-Augmented Generation (RAG) assistant that answers questions from a local text document (`data/sample_file.txt`) using Google Gemini.

It supports conversational querying, semantic retrieval, and context-aware follow-up questions.

---

## 🔧 Tech Stack

- **LLM:** `gemini-3-flash-preview`
- **Embeddings:** `models/gemini-embedding-2`
- **Vector Store:** FAISS
- **Framework:** LangChain
- **Language:** Python

---

## 📁 Project Structure

```

Crown Stack/
│── app.py
│── config.py
│── requirements.txt
│── README.md
│── data/
│   └── sample_file.txt
│── src/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── memory.py
│   ├── qa_chain.py
│   └── utils.py

````

---

## ⚙️ Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
````

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Add your Gemini API key
```
GEMINI_API_KEY=your_api_key_here
```
in "config.py"

## ▶️ Run the App

```powershell
python app.py
```
---

## 🧠 How It Works (RAG Pipeline)

1. **Document Loading**

   * The input text file is split into logical sections using heading-based parsing.
   * Each section is converted into a LangChain `Document`.

2. **Chunking**

   * Sections are further split using `RecursiveCharacterTextSplitter`
   * Chunking is tuned to preserve semantic meaning and improve retrieval accuracy

3. **Embedding**

   * Each chunk is converted into vector embeddings using `models/gemini-embedding-2`

4. **Vector Storage**

   * Embeddings are stored locally using FAISS

5. **Retrieval**

   * Top-k relevant chunks are retrieved based on semantic similarity

6. **LLM Response**

   * Retrieved context + user query is passed to `gemini-3-flash-preview`

7. **Memory**

   * Conversation history is maintained using `ConversationBufferMemory`
   * Enables follow-up questions and contextual understanding

---

## 💡 Key Features

* ✅ Conversational Q&A with memory
* ✅ Section-aware document ingestion (improves retrieval accuracy)
* ✅ Optimized chunking strategy
* ✅ Semantic search using FAISS
* ✅ Context-aware follow-up questions
* ✅ CLI-based interactive assistant

---

## 🧪 Example Queries

* What is the remote work policy?
* How many days can employees work remotely?
* Do employees need approval for remote work?
* What benefits are provided by the company?
* How much is the home office stipend?
* Compare the benefits with the leave policy
* What security policies are mentioned?

---

## ⚠️ Important Notes

* The FAISS index is stored in `faiss_index/`
* Delete `faiss_index/` if you change:

  * the document
  * chunk size
  * embedding model
* Ensure your Gemini API is enabled in your Google project
* API keys should be stored securely using `.env`

---

## 🚀 Future Improvements

* Web UI (Streamlit / React)
* Multi-document support
* Persistent memory storage
* Advanced retrieval (reranking / hybrid search)
* Evaluation metrics for answer quality

---

## 🧠 What This Project Demonstrates

* Understanding of RAG architecture
* LangChain integration (documents, retrievers, memory)
* Vector search using FAISS
* Context-aware question answering
* Debugging and optimization of chunking & retrieval

---

## 📌 Summary

This project implements a complete end-to-end RAG pipeline with conversational memory, structured document ingestion, and optimized retrieval using LangChain and Google Gemini.

```
