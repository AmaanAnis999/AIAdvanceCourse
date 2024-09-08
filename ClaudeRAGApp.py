# mkdir rag_app
# cd rag_app
# python -m venv venv
# source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# pip install transformers faiss-cpu sentence-transformers pypdf2 openai python-dotenv streamlit

import os
from PyPDF2 import PdfReader
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        end = start + chunk_size
    return chunks

def process_pdfs(directory: str) -> List[str]:
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text_into_chunks(text)
            all_chunks.extend(chunks)
    return all_chunks

# Usage
pdf_directory = 'path/to/your/pdfs'
text_chunks = process_pdfs(pdf_directory)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    return model.encode(texts)

import faiss
import numpy as np

def create_vector_db(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Assuming text_chunks is your list of text chunks
embeddings = get_embeddings(text_chunks)
vector_db = create_vector_db(embeddings)

import numpy as np

def retrieve_relevant_chunks(query, vector_db, text_chunks, k=5):
    query_embedding = get_embeddings([query])[0]
    distances, indices = vector_db.search(np.array([query_embedding]), k)
    
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks

def create_prompt(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"""Given the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
    return prompt

# OPENAI_API_KEY=your_api_key_here make this in .env file

import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

import streamlit as st
from pdf_processor import process_pdfs
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from retrieval_mechanism import retrieve_relevant_chunks, create_prompt
from openai_integration import generate_response

# Initialize the app
st.title("RAG Application")

# Load and process PDFs
pdf_directory = st.text_input("Enter the path to your PDF directory:")
if pdf_directory:
    text_chunks = process_pdfs(pdf_directory)
    st.success(f"Processed {len(text_chunks)} text chunks from PDFs.")

    # Create embeddings and vector database
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    dimension = embeddings.shape[1]
    vector_db = faiss.IndexFlatL2(dimension)
    vector_db.add(embeddings)

    # User input
    query = st.text_input("Enter your question:")
    if query:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(query, vector_db, text_chunks)

        # Create prompt and generate response
        prompt = create_prompt(query, relevant_chunks)
        response = generate_response(prompt)

        # Display response
        st.write("Response:", response)

# Run the app
if __name__ == "__main__":
    st.run()


# streamlit run app.py run this in terminal