# # !pip install qdrant_client langchain_huggingface langchain-community langchain-qdrant pypdf openai langchain transformers langchain_huggingface

# from qdrant_client import QdrantClient
# from langchain_core.documents import Document
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_qdrant import QdrantVectorStore
# import openai
# import os

# # Initialize embedding model with BAAI/bge-small-en-v1.5
# embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# # Load the PDF document using PyPDFLoader
# loaders = PyPDFLoader("/content/National_AI_Policy_Consultation_Draft_1722220582.pdf")


# # Extract pages from the loaded PDF
# pages = loaders.load()

# pages[15]

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1500,
#     chunk_overlap = 150
# )

# # Create an empty list to store processed document chunks
# doc_list = []

# # Iterate over each page in the extracted pages
# for page in pages:
#     # Split the page content into smaller chunks
#     pg_split = text_splitter.split_text(page.page_content)

#     # Iterate over each chunk and create Document objects
#     for pg_sub_split in pg_split:
#         # Metadata for each chunk, including source and page number
#         metadata = {"source": "AI policy", "page_no": page.metadata["page"] + 1}

#         # Create a Document object with content and metadata
#         doc_string = Document(page_content=pg_sub_split, metadata=metadata)

#         # Append the Document object to the list
#         doc_list.append(doc_string)

# qdrant_url = ""
# qdrant_key = ""
# collection_name = "AI_policy_new"

# # Initialize QdrantVectorStore with documents and embedding model
# qdrant = QdrantVectorStore.from_documents(
#     doc_list,                # List of Document objects to be stored in the vector store
#     embed_model,             # Embedding model used to convert documents into vectors
#     url=qdrant_url,          # URL for the Qdrant service
#     api_key=qdrant_key,      # API key for accessing the Qdrant service
#     collection_name=collection_name  # Name of the collection to store the vectors in
# )

# question = "What is AI policy for students?"

# docs_ss = qdrant.similarity_search(question,k=5)

# docs_ss[0].page_content

# docs_ss[1].page_content

# docs_mmr = qdrant.max_marginal_relevance_search(question,k=5)

# docs_mmr[0].page_content

# docs_mmr[1].page_content

# docs = qdrant.similarity_search(
#     question,
#     k=3,
#     filter={
#         "must": [
#             {
#                 "key": "page",
#                 "match": {
#                     "value": 3
#                 }
#             }
#         ]
#     }
# )

# docs

# from langchain.retrievers import SVMRetriever
# from langchain.retrievers import TFIDFRetriever
# # from langchain.document_loaders import PyPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter

# texts = [doc.page_content for doc in doc_list]
# svm_retriever = SVMRetriever.from_texts(texts, embed_model)
# tfidf_retriever = TFIDFRetriever.from_texts(texts)

# question = "what is AI policy document about?"
# docs_svm=svm_retriever.get_relevant_documents(question)
# docs_svm[0]

# question = "what is AI policy document about?"
# docs_tfidf=tfidf_retriever.get_relevant_documents(question)
# docs_tfidf[0]


# UNCOMMENT THIS FILE ONCE AND WORK