import os
import pickle
import re

import faiss
import google.generativeai as genai

import numpy as np
import PyPDF2
from dotenv import load_dotenv

# Load env variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing API key! Set GOOGLE_API_KEY in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# Load the PDF and extract text
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Split the documents into chunks
def chunk_text(text):
    # Split by sections (SEC. <section number>.)
    chunks = re.findall(r'SEC\. \d+\..*?(?=SEC\. \d+\.|$)', text, re.DOTALL)
    # Remove empty strings from the list
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks

# Generate embeddings and store in FAISS
def create_faiss_index(model_name, faiss_index_path, chunks_path, chunks):
    embeddings = [
        genai.embed_content(
            model=model_name,
            content=chunk,
            task_type="retrieval_document"
        )["embedding"]
        for chunk in chunks
    ]

    embeddings_array = np.array(embeddings, dtype=np.float32)
    embedding_dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)

    # Save FAISS index
    faiss.write_index(index, faiss_index_path)

    # Save chunks
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

    return index

# Perform semantic search
def search_faiss(model_name, query, chunks, index, top_k=3):
    query_embedding = genai.embed_content(
        model=model_name,
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    distances, indices = index.search(
        np.array([query_embedding], dtype=np.float32),
        top_k
    )
    return [chunks[i] for i in indices[0]]

# Generate response
def generate_response(model, query, context):
    prompt = f"""You are a legal expert. Use the following context to answer the question. Do not hint that you're using context.
    Context: {context}
    Question: {query}"""
    response = model.generate_content(prompt)
    return response.text

# Main
pdf_path = "philippines-2019-legislation-ra-11232-revised-corporation-code-2019.pdf"
faiss_index_path = "faiss.index"
chunks_path = "faiss_index.pkl"
model_name = "models/text-embedding-004"
llm_name = "gemini-1.5-pro-latest"

# Load the vector store if it exists, otherwise create it
if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
    print("Loading existing FAISS index...")

    # Load the FAISS index
    index = faiss.read_index(faiss_index_path)
    
    # Load the chunks
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
else:
    print("Creating new FAISS index...")

    # Extract text, chunk, and create the FAISS index
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    index = create_faiss_index(model_name, faiss_index_path, chunks_path, chunks)    

# LLM
llm = genai.GenerativeModel(llm_name)

# Interactive search
while True:
    query = input("Ask a legal question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    try:
        # 1. Retrieve
        relevant_info = search_faiss(model_name, query, chunks, index)

        # 2. Augment
        context = "\n".join(relevant_info)

        # 3. Generate
        response = generate_response(llm, query, context)

        print("\nGemini's Answer:")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")        
