# Vector Store
# One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. A vector store takes care of storing embedded data and performing vector search for you.

# !pip install qdrant_client langchain_huggingface langchain-community langchain-qdrant pypdf openai langchain transformers langchain_huggingface

from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import openai
import os

# Initialize embedding model with BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')


# Load the PDF document using PyPDFLoader
loaders = PyPDFLoader("plant.pdf")

# Extract pages from the loaded PDF
pages = loaders.load()

pages[15]

len(pages)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

from langchain.docstore.document import Document

# Create an empty list to store processed document chunks
doc_list = []

# Iterate over each page in the extracted pages
for page in pages:
    # Split the page content into smaller chunks
    pg_split = text_splitter.split_text(page.page_content)

    # Iterate over each chunk and create Document objects
    for pg_sub_split in pg_split:
        # Metadata for each chunk, including source and page number
        metadata = {"source": "AI policy", "page_no": page.metadata["page"] + 1}

        # Create a Document object with content and metadata
        doc_string = Document(page_content=pg_sub_split, metadata=metadata)

        # Append the Document object to the list
        doc_list.append(doc_string)

doc_list[10]

len(doc_list)

# Qdrant Vectore Store
# Qdrant Credentials
qdrant_url = "https://bd4e1587-ec02-453f-b686-77a9b8be1b0c.europe-west3-0.gcp.cloud.qdrant.io:6333"
qdrant_key = ""
collection_name = "AI_policy_new"

# Initialize QdrantVectorStore with documents and embedding model
qdrant = QdrantVectorStore.from_documents(
    doc_list,                # List of Document objects to be stored in the vector store
    embed_model,             # Embedding model used to convert documents into vectors
    url=qdrant_url,          # URL for the Qdrant service
    api_key=qdrant_key,      # API key for accessing the Qdrant service
    collection_name=collection_name  # Name of the collection to store the vectors in
)

# Query Vector Store
# Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

# Query directly
# The simplest scenario for using Qdrant vector store is to perform a similarity search. Under the hood, our query will be encoded into vector embeddings and used to find similar documents in Qdrant collection.

query = "what is Ai policy for students?"

# Retrieve relevant documents
results = qdrant.similarity_search(query, k=5)

results[3]

results[0].page_content

# Pinecone Vector Store
# %pip install -qU langchain-pinecone pinecone-notebooks

PINECONE_API_KEY=""
index_name="demo-vectorstore"

from langchain_pinecone import PineconeVectorStore as lang_pinecone
import os
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Convert documents into vectors using LangPinecone
vector = lang_pinecone.from_documents(
    doc_list,                # List of Document objects to be converted into vectors
    embed_model,             # Embedding model used for generating vector representations
    index_name=index_name    # Name of the Pinecone index where vectors will be stored
)

# Define a query to search for relevant information
query = "What is AI policy for students?"

# Perform similarity search to find the top 5 most relevant results
pinecone_results = vector.similarity_search(query, k=5)

pinecone_results