from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

load_dotenv()

# change the path accordingly
directory = "Content"
loader = PyPDFDirectoryLoader(directory)    
documents = loader.load()
print('Loading of doc done!')

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

try:
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
except RuntimeError:
    print("Cuda not available. Using CPU instead.")
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

model_dimensions = 384

pc = Pinecone()
p_index = os.getenv('pinecone_index_name')
if p_index in pc.list_indexes().names():
    print("Deleted the Existing Index")
    pc.delete_index(p_index)

print("Creating the index")
pc.create_index(
    name=p_index,
    dimension=model_dimensions,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# The storage layer for the parent documents
store = InMemoryStore()

namespace = "studymate_chatbot"
vectorstore = PineconeVectorStore(
    index_name=p_index,
    embedding=hf,
    namespace=namespace,
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)
print("All the documents are uploaded in the vectorestore!!")