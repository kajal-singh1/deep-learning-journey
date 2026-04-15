# 📚 Complete RAG Pipeline using LangChain + ChromaDB + LLMs

## 🚀 Overview

This project implements a **full Retrieval-Augmented Generation (RAG) system** that:

✔ Loads **Text + PDF documents**  
✔ Splits documents into chunks  
✔ Converts text into embeddings using Sentence Transformers  
✔ Stores embeddings in **ChromaDB vector database**  
✔ Retrieves relevant chunks using semantic search  
✔ Generates final answers using **LLMs (OpenAI / Groq)**  

---

## 📂 Project Structure

```
rag/
│── data/
│   ├── pdfs/                 # Input PDF files
│   ├── Python.txt            # Sample text file
│   ├── vector_store/         # Persistent ChromaDB storage
│
│── RAG_pipeline.ipynb        # Main notebook
│── README.md
│── .gitignore
```

---

## ⚙️ Tech Stack

- LangChain  
- ChromaDB (Persistent Vector Database)  
- Sentence Transformers (`all-MiniLM-L6-v2`)  
- PyPDFLoader & PyMuPDFLoader  
- Scikit-learn (Cosine Similarity)  
- OpenAI / Groq LLMs  

---

## 🔄 End-to-End Pipeline Flow

```
Text/PDF → Documents → Chunks → Embeddings → Vector DB → Query → Retrieval → LLM Response
```

---

## 🧩 Pipeline Components

### 1. Data Loading

#### 📄 Text Loader
- Loads `.txt` files using `TextLoader`

#### 📑 PDF Loader
- Loads multiple PDFs from:
```
data/pdfs/
```
- Uses:
  - `PyPDFLoader`
  - (Optional) `PyMuPDFLoader`

---

### 2. Document Processing

- Converts raw data → LangChain `Document` objects  
- Extracts:
  - `page_content`
  - `metadata`

---

### 3. Text Chunking

- Uses `RecursiveCharacterTextSplitter`  
- Default:
  - Chunk size = 500  
  - Overlap = 50  

---

### 4. Embedding Generation

Custom class: `EmbeddingManager`

- Model: `all-MiniLM-L6-v2`  
- Converts text → vector embeddings  

---

### 5. Vector Store (ChromaDB)

Custom class: `VectorStoreManager`

- Storage path:
```
data/vector_store/
```

Stores:
- IDs (UUID)  
- Text  
- Metadata  
- Embeddings  

---

### 6. Retrieval System

Custom class: `RAGRetriever`

Steps:
1. Query → embedding  
2. Semantic search in DB  
3. Similarity score:
```
similarity = 1 - distance
```
4. Return top results  

---

### 7. LLM Integration

#### 🤖 OpenAI
- `ChatOpenAI`
- Model: `gpt-5.4`

#### ⚡ Groq
- `ChatGroq`
- Model: `qwen/qwen3-32b`

---

### 8. Final RAG Generation

Function: `generate_output()`

Steps:
1. Retrieve docs  
2. Build context  
3. Combine:
```
Context + Query
```
4. Send to LLM  
5. Get final answer  

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install langchain langchain-core langchain-community \
pypdf pymupdf sentence-transformers chromadb \
langchain-openai langchain-groq langchain-text-splitters scikit-learn
```

---

### 2. Add Data

- PDFs → `data/pdfs/`  
- Text → `data/Python.txt`  

---

### 3. Run

Open:
```
RAG_pipeline.ipynb
```

---

### 4. Example Usage

#### Retrieval
```python
rag_retriever.retrieve("What is encoder decoder")
```

#### Full RAG
```python
generate_output("What is RAG?", rag_retriever, llm)
```

---

## 📊 Sample Output

```json
[
  {
    "id": "doc_xxx",
    "document": "...",
    "metadata": {...},
    "similarity_score": 0.87,
    "rank": 1
  }
]
```

---

## ⚠️ Notes

- Ignore `data/` and `vector_store/` in `.gitignore`  
- Works best with clean PDFs  
- Embeddings run locally  
- ChromaDB persists automatically  

---

## 💡 Future Improvements

- Streamlit / Gradio UI  
- Chat memory  
- Hybrid search  
- Reranking  
- Multi-modal support  

---

## 🧠 Learning Outcomes

- RAG pipeline design  
- Vector databases  
- Embeddings  
- LLM integration  
- End-to-end AI system  

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub 🚀