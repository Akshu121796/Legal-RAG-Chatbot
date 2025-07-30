# 🧠 Local Legal RAG Chatbot (Chainlit + LLaMA + FAISS)

A local Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and ask questions about its content — all without any OpenAI or cloud dependencies.

## ✅ Features
- 🔍 Context-aware answers from uploaded PDF
- 📁 Drag & drop PDF support
- 🧠 Local LLM via `llama-cpp-python`
- 📦 Uses HuggingFace sentence-transformers for embedding
- 🚫 No internet required once set up

---

### 📁Project structure
legal-rag-chatbot/
│
├── app.py                      
├── requirements.txt            
├── README.md                   
├── .gitignore                  
├── models/                      
│   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
├── db/                        
│   └── index/                   
├── temp/                        
└── assets/ (optional)        
    └── demo.png

## 🛠 Setup Instructions

###
```bash
git clone https://github.com/Akshu121796/legal-rag-chatbot.git
cd legal-rag-chatbot
```

### 2. Download the LLM MOdel
```bash
Visit: mistral-7b-instruct-v0.1 GGUF (Q4_K_M)
Download:" mistral-7b-instruct-v0.1.Q4_K_M.gguf"(place in models/)
```

### 3. Create and activate virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  ( on Windows use venv\Scripts\activate)
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Chatbot Locally
```bash
chainlit run app.py 
```


### 📝 Note
- If the model is too large for your system, use a smaller quantized .gguf variant.

- Use only legal and public PDFs for testing.

## 🤝 Contributing
Feel free to fork the repo and submit pull requests.

