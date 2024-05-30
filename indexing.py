from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv

load_dotenv()

# change the path accordingly
directory = "Content"
loader = PyPDFDirectoryLoader(directory)
documents = loader.load()
print('Loading of doc done!')

def split_docs(documents, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs

print("Total Splits: ",len(documents))
docs = split_docs(documents)
print("Total Splits: ",len(docs))

# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
# index_name = "chatbot"
# docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
# index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)