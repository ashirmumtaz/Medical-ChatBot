from src.helper import load_pdf_file,text_split,download_HuggingFace_Embeddings
import pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]="78d6dda4-ba2b-412e-9ebe-64c5a3fd3e27"


extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_HuggingFace_Embeddings()

pc= Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


docsearch = PineconeVectorStore.from_documents(
    documents= text_chunks,
    index_name=index_name,
    embedding=embeddings
)
