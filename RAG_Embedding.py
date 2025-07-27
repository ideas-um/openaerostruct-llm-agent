import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_together import TogetherEmbeddings

file = "./RAG_PDF/Aircraft_Design_Metabook.pdf"
path = os.path.abspath(file)

loader = PyPDFLoader(path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True, length_function=len
)
all_splits = text_splitter.split_documents(docs)

# The splits are already Document objects
documents = all_splits

# Import the embeddings
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-32k-retrieval",
    api_key= os.environ.get('TOGETHER_AI_API_KEY')
)

# Chroma is the vector database
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='./Embeddings_Persist'
)

vector_store.add_documents(documents=documents)