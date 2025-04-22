import os
from dotenv import load_dotenv

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load env variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing API key! Set GOOGLE_API_KEY in your .env file.")

# Load the PDF and extract text
def extract_documents_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Split the documents into chunks
def chunk_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Generate embeddings and store in FAISS
def create_faiss_index(model, faiss_index_path, chunks):
    index = FAISS.from_documents(chunks, model)
    # index = FAISS.from_documents(chunks, model, index_factory="HNSW32")
    index.save_local(faiss_index_path)
    return index

# Main
pdf_path = "philippines-2019-legislation-ra-11232-revised-corporation-code-2019.pdf"
faiss_index_path = "faiss.index"
model_name = "models/text-embedding-004"
llm_name = 'gemini-1.5-pro-latest' # 'gemini-pro' or 'gemini-1.5-pro-latest'

# Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=GOOGLE_API_KEY)

# Load the vector store if it exists, otherwise create it
if os.path.exists(faiss_index_path):
    print("Loading existing FAISS index...")

    index = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS index...")

    # Extract text, chunk, and create the FAISS index
    documents = extract_documents_from_pdf(pdf_path)
    chunks = chunk_text(documents)
    index = create_faiss_index(embedding_model, faiss_index_path, chunks)

# LLM
llm = ChatGoogleGenerativeAI(model=llm_name, google_api_key=GOOGLE_API_KEY, temperature=0.3, callbacks=[StreamingStdOutCallbackHandler()])
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.as_retriever(),
    chain_type_kwargs={"prompt": PromptTemplate(input_variables=["context", "question"], template="You are a legal expert. Use the context below to answer the question clearly and concisely.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")}
)

# Interactive search
while True:
    query = input("Ask a legal question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    try:
        response = qa.invoke(query)
        
        print("\nGemini's Answer:")
        print(response["result"])
    except Exception as e:
        print(f"An error occurred: {e}")