#Analyze the results with RAG of the Martins Metabook#
from langchain_chroma import Chroma
from langchain_together import TogetherEmbeddings
import os

# Import the embeddings
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-32k-retrieval",
    api_key= os.environ.get('TOGETHER_AI_API_KEY')
)

persist_directory = './Embeddings_Persist'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever(search_type="similarity",
    search_kwargs={'k': 5})

def run_retriever(query):
    output = retriever.invoke(query)
    return output