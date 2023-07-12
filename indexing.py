from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

# change the path accordingly
directory = 'D:\Chatbot\Content'
loader = PyPDFDirectoryLoader(directory)
documents = loader.load()
# print('Loading of doc done!')

def split_docs(documents,chunk_size=1000,chunk_overlap=150):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
# print(len(docs))

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

pinecone.init(
    api_key = PINECONE_API_KEY,
    environment = PINECONE_ENVIRONMENT
)
index_name = "chatbot"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# function to provide the gpt with context of the given docs
def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs