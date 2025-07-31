import os
import chainlit as cl
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
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = None
llm = None

# Utilities
def extract_text_from_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

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
    return f"""You are a legal assistant AI. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

def load_llm():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return LlamaCpp(
        model_path=MODEL_PATH,
    temperature=0.5,
    max_tokens=512,
    top_p=1,
    n_ctx=1024,
    n_threads=os.cpu_count(),
    n_batch=128,
    streaming=True,
    verbose=False
)
       

# Chainlit lifecycle
@cl.on_chat_start
async def on_chat_start():
    global llm, vectorstore
    llm = load_llm()
    await cl.Message(content="👋 Welcome to the Legal RAG Chatbot!\n\n").send()

    # If vector index doesn't exist, ask for PDF
    if not os.path.exists(VECTOR_DB_PATH):
        await cl.Message(content="📄 Please upload a legal PDF to get started.").send()
        upload_msg = cl.AskFileMessage(
            content="Upload a PDF file to build context.",
            accept=["application/pdf"],
            max_size_mb=10
        )
        uploaded = await upload_msg.send()
        file = uploaded[0]
        file_path = file.path 


        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)
        vectorstore = build_index(chunks)
        await cl.Message(content="✅ PDF indexed. Ask your legal questions!").send()
    else:
        vectorstore = load_index()
        await cl.Message(content="📚 Existing index loaded. Ask your legal questions!").send()


@cl.on_message
async def handle_message(message):
    if any(q in message.content.lower() for q in ["who are you", "tell me about yourself", "about you", "your purpose"]):
        await cl.Message(
            content="""
🤖 I’m a Legal RAG Chatbot built by **Akshata Chettiar** to help answer questions based on uploaded legal documents like contracts or policies.

🔗 You can connect with Akshata on [LinkedIn](https://www.linkedin.com/in/akshata-chettiar)

Just upload a PDF and ask your legal queries!
"""
        ).send()
        return
        
    if not vectorstore:
        await cl.Message(content="❌ Please upload a PDF first.").send()
        return

    query = message.content
    relevant_chunks = search_index(query)

    if not relevant_chunks:
        await cl.Message(content="⚠️ Couldn't find relevant info in the PDF. Try rephrasing.").send()
        return

    prompt = format_prompt(query, relevant_chunks)

    response = await cl.make_async(llm)(prompt)
    await cl.Message(content=response.strip()).send()
