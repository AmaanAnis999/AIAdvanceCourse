from google.colab import drive
drive.mount('/content/drive')

# !pip install qdrant_client langchain_huggingface langchain-community langchain-qdrant pypdf openai langchain transformers langchain_huggingface langchain_openai

import openai

openai.api_key = ""
qdrant_url = ""
qdrant_key = ""
collection_name = "ML_Lectures"
llm_name = "gpt-4o-mini"

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
prompt_str="""
Answer the user question briefly.

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_str)
question_fetcher=itemgetter("question")
llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai.api_key, temperature=0)

chain = question_fetcher| prompt | llm | StrOutputParser()
query = "tell me about lahore"  # Question here
response = chain.invoke({"question": query})
print(response)

from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import openai
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


loaders = PyPDFLoader("/content/Dsa.pdf")
pages = loaders.load()

len(pages)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(pages)

len(splits)

qdrant = QdrantVectorStore.from_documents(
    splits,
    embed_model,
    url=qdrant_url,
    prefer_grpc=True,
    api_key=qdrant_key,
    collection_name=collection_name,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_str="""
Answer the user question based only on the following context:
{context}

Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)
num_chunks=3
retriever = qdrant.as_retriever(search_type="similarity",
                                        search_kwargs={"k": num_chunks})
chat_llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai.api_key, temperature=0)
query_fetcher= itemgetter("question")
setup={"question":query_fetcher,"context":query_fetcher | retriever | format_docs}
_chain = (setup |_prompt | chat_llm)


query="what is DSA?"

response=_chain.invoke({"question":query})

response

response.content

history = []


prompt_str="""
Answer the user question briefly.

Question: {question}

conversation_history: {chat_history}
"""
prompt = ChatPromptTemplate.from_template(prompt_str)
question_fetcher=itemgetter("question")
history_fetcher=itemgetter("chat_history")
llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai.api_key, temperature=0)
setup={"question":question_fetcher,"chat_history":history_fetcher}
chain = setup|prompt | llm | StrOutputParser()
query = "tell me about lahore"
response = chain.invoke({"question": query,"chat_history":"\n".join(str(history))})
print(response)
query="user_question:"+query
response="ai_response:"+response
history.append((query, response))


history