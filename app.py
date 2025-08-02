import os
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp

# Constants
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "db/index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Globals
llm = None
vectorstore = None
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Utilities
def extract_text_from_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def build_index(chunks):
    vs = FAISS.from_texts(chunks, embedding_model)
    vs.save_local(VECTOR_DB_PATH)
    return vs

def load_index():
    return FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

def search_index(query):
    docs = vectorstore.similarity_search(query, k=3)
    return [doc.page_content for doc in docs]

def format_prompt(question, context_chunks):
    context = "\n".join(context_chunks)
    return f"""You are a helpful legal assistant AI.

Context:
{context}

Question: {question}
Answer:"""

def load_llm():
    if not os.path.exists(MODEL_PATH):
        return None
    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        n_ctx=2048,
        n_threads=os.cpu_count(),
        verbose=True
    )

# Chatbot logic
def process_pdf_and_query(pdf, query):
    global llm, vectorstore

    if not pdf:
        return "❌ Please upload a PDF."

    # Save PDF locally
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{pdf.name}"
    with open(file_path, "wb") as f:
        f.write(pdf.read())

    # Process PDF
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    vectorstore = build_index(chunks)

    # Load model
    llm = load_llm()
    if not llm:
        return "⚠️ Model file missing. Please add your `.gguf` model to the `models/` folder."

    # Run query
    relevant_chunks = search_index(query)
    prompt = format_prompt(query, relevant_chunks)
    response = llm(prompt)

    return response

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧑‍⚖️ Legal RAG Chatbot (Local Mistral + HuggingFace Embeddings)")
    with gr.Row():
        pdf_input = gr.File(label="Upload Legal PDF", file_types=[".pdf"])
        query_input = gr.Textbox(label="Ask a question", placeholder="What are the key clauses?")
    output = gr.Textbox(label="Response")

    submit_btn = gr.Button("Ask")

    submit_btn.click(fn=process_pdf_and_query, inputs=[pdf_input, query_input], outputs=output)

# Launch app
demo.launch()
