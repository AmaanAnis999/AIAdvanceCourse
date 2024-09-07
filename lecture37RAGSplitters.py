# pip install openai langchain
import os
import openai
import sys
sys.path.append('../..')

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

#do some experiemnts with different chunk size and overlap
chunk_size =26
chunk_overlap = 4

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    # separators=["\n","\n\n"," ",""],
    chunk_overlap=chunk_overlap
)

c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    # separator="\n\n",
    chunk_overlap=chunk_overlap
)

text1 = 'abcdefghijklmnopqrstuvwxyz'

text1

r_splitter.split_text(text1)

c_splitter.split_text(text1)

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'

r_splitter.split_text(text2)

c_splitter.split_text(text2)

# %pip install --upgrade --quiet langchain-text-splitters tiktoken

t = "LangChain was launched in October 2022 as an open source project by Harrison Chase, while working at machine learning startup Robust Intelligence. The project quickly garnered popularity,with improvements from hundreds of contributors on GitHub, trending discussions on Twitter, lively activity on the project's Discord server, many YouTube tutorials, and meetups in San Francisco and London. In April 2023, LangChain had incorporated and the new startup raised over $20 million in funding at a valuation of at least $200 million from venture firm Sequoia Capital, a week after announcing a $10 million seed investment from Benchmark."

# %pip install --upgrade --quiet  spacy

from langchain_text_splitters import SpacyTextSplitter

text_splitter = SpacyTextSplitter(chunk_size=10, chunk_overlap=5)
texts = text_splitter.split_text(t)

texts[0]

# !pip install pypdf langchain_community

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/National_AI_Policy_Consultation_Draft_1722220582 (1).pdf")
pages = loader.load_and_split()

len(pages)

pages[20]

chunk_list=[]
for page in pages:
  chunks=r_splitter.split_text(page.page_content)
  for chunk in chunks:
    chunk_list.append(chunk)
print(len(chunk_list))

chunk_list[1000]

# !pip install transformers langchain_huggingface

from langchain_huggingface import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

embed = embed_model.embed_query("how are you?")

len(embed)

embed

emb  = embed_model.embed_query(chunk_list[3])

len(emb)

emb

from langchain_openai import OpenAIEmbeddings
embed_fn = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

embed = embed_fn.embed_query("how are you?")

# !pip install langchain_experimental

from langchain_experimental.text_splitter import SemanticChunker

# !pip install --quiet langchain_experimental langchain_openai

text_splitter = SemanticChunker(
    embed_model, breakpoint_threshold_type="percentile"
)

t

sementic_splits = text_splitter.split_text(t)

len(sementic_splits)

sementic_splits[0]

sementic_splits[1]

sementic_embed = embed_model.embed_query(sementic_splits[0])

sementic_embed

len(sementic_embed)