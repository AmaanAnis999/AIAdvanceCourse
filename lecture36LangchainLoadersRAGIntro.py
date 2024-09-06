# pip install langchain
# pip install openai
# pip install python-dotenv
# pip install langchain langchain-community

import os
import openai
import sys
sys.path.append('../..')

# PDF LOADER
# pip install pypdf
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("plantnest.pdf") #add path of your document
pages = loader.load()

len(pages)

page = pages[0]

page

print(page.page_content[0:500])

page.metadata['page']

# TEXT LOADER
from langchain_community.document_loaders import TextLoader

loader = TextLoader("/content/National AI policy.txt")
data = loader.load()

len(data)

data

# DOCS LOADER
# !pip install docx2txt
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("/content/basic course outline.docx") # add my docs
doc_data = loader.load()

doc_data

doc_data[0].metadata

# WEBBASELOADER / URL LOADER
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.xevensolutions.com/")

docs = loader.load()

docs

print(docs[0].page_content[:500])