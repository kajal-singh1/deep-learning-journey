# 📚 RAG Pipeline using LangChain + ChromaDB

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that:

* Loads PDF documents
* Splits them into chunks
* Converts text into embeddings
* Stores embeddings in a vector database
* Retrieves relevant documents based on user queries

---

## 📂 Project Structure

```
rag/
│── data/
│   ├── pdfs/                # Input PDF files
│── vector_store/        # ChromaDB storage
│
│── RAG_pipeline.ipynb       # Main notebook
│── README.md
│── .gitignore
```

---

## ⚙️ Tech Stack

* LangChain
* ChromaDB (Vector Database)
* Sentence Transformers
* PyPDF Loader
* Scikit-learn

---

## 🔄 Pipeline Flow

### 1. Data Ingestion

* Load PDFs using `PyPDFLoader`
* Convert into LangChain `Document` objects

---

### 2. Text Splitting

* Uses `RecursiveCharacterTextSplitter`
* Breaks large text into smaller chunks

---

### 3. Embedding Generation

* Model: `all-MiniLM-L6-v2`
* Converts text → numerical vectors

---

### 4. Vector Storage

* Uses ChromaDB
* Stores:

  * IDs
  * Embeddings
  * Documents
  * Metadata

---

### 5. Retrieval

* Query → embedding
* Semantic search in vector DB
* Returns top relevant chunks

---

## 🧠 How It Works

```
PDF → Text → Chunks → Embeddings → Vector DB → Query → Retrieved Docs
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install langchain langchain-community pypdf pymupdf sentence-transformers chromadb
```

---

### 2. Add PDFs

Place your PDF files inside:

```
data/pdfs/
```

---

### 3. Run Notebook

Open and run:

```
RAG_pipeline.ipynb
```

---

### 4. Query Example

```python
rag_retriever.retrieve("What is encoder decoder")
```

---

## 📊 Output Example

```json
[
  {
    "id": "doc_xxx",
    "document": "...",
    "metadata": {...},
    "similarity_score": 0.85,
    "rank": 1
  }
]
```

---

## ⚠️ Notes

* `data/` and `vector_store/` are ignored in `.gitignore`
* Embeddings are regenerated when needed
* Works best with clean PDF text

---

## 💡 Future Improvements

* Add LLM response generation (Chatbot)
* Use OpenAI / HuggingFace models
* Add UI (Streamlit / Gradio)
* Optimize chunking strategy
* Add hybrid search (keyword + vector)

---

## 🙌 Author

Built as part of learning **RAG systems and LLM pipelines**

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!
