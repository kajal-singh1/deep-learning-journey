# 📄 Text Summarizer App (T5 + FastAPI)

## 🚀 Overview
This project is a **Text Summarization Web App** built using:
- **T5 Transformer Model**
- **FastAPI (Backend)**
- **Jinja2 Templates (Frontend)**

It summarizes long dialogues into short, meaningful summaries using NLP.

---

## 📁 Project Structure
```
.
├── Text_Summarizer_App/
│   ├── saved_summary_model/   # Fine-tuned model (ignored in git)
│   ├── t5env/                 # Virtual environment (ignored)
│   ├── templates/             # HTML templates
│   └── app.py                 # Main FastAPI app
│
├── samsum-train.csv           # Training dataset
├── samsum-test.csv            # Test dataset
├── text_summarizer.ipynb      # Model training notebook
├── .gitignore
└── README.md
```

---

## ⚙️ Features
- ✅ Dialogue summarization using **T5-small**
- ✅ FastAPI REST API
- ✅ Simple web interface (HTML templates)
- ✅ Supports GPU (CUDA / MPS) if available
- ✅ Text preprocessing for better results

---

## 🧠 Model Training
- Dataset: **SAMSum (dialogue summarization dataset)**
- Fine-tuning done using:
  - `T5ForConditionalGeneration`
  - HuggingFace `Trainer API`
- Training notebook: `text_summarizer.ipynb`

---

## 🛠️ Installation

### 1. Clone Repository
```
git clone <your-repo-url>
cd <repo-name>
```

### 2. Create Virtual Environment
```
python -m venv t5env
```

### 3. Activate Environment

**Windows:**
```
t5env\Scripts\activate
```

**Mac/Linux:**
```
source t5env/bin/activate
```

### 4. Install Dependencies
```
pip install fastapi uvicorn transformers torch jinja2
```

---

## ▶️ Run the App
```
cd Text_Summarizer_App
uvicorn app:app --reload
```

Open in browser:
```
http://127.0.0.1:8000
```

---

## 📡 API Usage

### POST `/summarize/`

**Request:**
```json
{
  "dialogue": "Your long text here..."
}
```

**Response:**
```json
{
  "summary": "Short summarized text"
}
```

---

## 🔍 How It Works
1. Input dialogue is cleaned (remove extra spaces, HTML, etc.)
2. Tokenized using T5 tokenizer
3. Passed into T5 model
4. Summary generated using beam search
5. Output decoded into readable text

---

## 📌 Notes
- `.csv` files and model files are ignored using `.gitignore`
- By default, app uses pretrained `t5-small`
- To use your fine-tuned model, update:
```
from_pretrained("./saved_summary_model")
```

---

## 🎯 Future Improvements
- Improve UI design
- Deploy on cloud (AWS / Render / HuggingFace)
- Use larger models (T5-base, FLAN-T5)
- Add evaluation metrics (ROUGE score)

---

## 👨‍💻 Author
- Kajal Singh