#1. Data Loading and Preprocessing
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf_reports(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            with open(os.path.join(folder_path, filename), 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    texts.append(page.extract_text())
    return texts

def split_texts(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(texts)

# Load and split the texts
folder_path = 'path/to/pdf/folder'  # Update this path
raw_texts = load_pdf_reports(folder_path)
segments = split_texts(raw_texts)









# 2. Embedding Creation and Storage
from sentence_transformers import SentenceTransformer
from chromadb import Client

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for each segment
embeddings = embedding_model.encode(segments)

# Initialize Chroma client and store embeddings
chroma_client = Client()  # Update this to your Chroma configuration
collection = chroma_client.create_collection('esg_reports')

# Store embeddings and metadata
for i, segment in enumerate(segments):
    metadata = {
        'company_name': 'Your Company Name',  # Modify as necessary
        'report_year': '2023'  # Modify as necessary
    }
    collection.add(ids=[f'doc_{i}'], embeddings=[embeddings[i]], metadatas=[metadata])











#3. Question-Answering and Retrieval
from langchain.chains import RetrievalQA
from langchain.llms import Mistral

# Initialize the Mistral model
mistral_model = Mistral()

# Set up the QA chain with retrieval
qa_chain = RetrievalQA(
    llm=mistral_model,
    retriever=collection.as_retriever(),  # Use the Chroma vector store retriever
)

# Sample query
query = "What are the key ESG factors discussed?"
response = qa_chain({"query": query})
print(response)




