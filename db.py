from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
from langchain_pinecone import PineconeVectorStore

load_dotenv()
p_index = os.getenv('pinecone_index_name')

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

namespace = "studymate_chatbot"
docsearch = PineconeVectorStore.from_documents(
    documents=[],
    index_name=p_index,
    embedding=hf, 
    namespace=namespace 
)

retriever = docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 4, 'lambda_mult': 0.25}
            )