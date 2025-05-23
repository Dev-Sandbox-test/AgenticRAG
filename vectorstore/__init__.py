import os
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    VECTOR_STORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def get_embeddings_model():
    """Initialize and return the Azure OpenAI embeddings model."""
    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def load_documents(data_dir: str) -> List:
    """Load documents from the specified directory."""
    loaders = {
        ".pdf": (DirectoryLoader, {"path": data_dir, "glob": "**/*.pdf", "loader_cls": PyPDFLoader}),
        ".txt": (DirectoryLoader, {"path": data_dir, "glob": "**/*.txt", "loader_cls": TextLoader}),
        ".docx": (DirectoryLoader, {"path": data_dir, "glob": "**/*.docx", "loader_cls": Docx2txtLoader}),
    }
    
    documents = []
    for file_type, (loader_class, loader_kwargs) in loaders.items():
        try:
            loader = loader_class(**loader_kwargs)
            documents.extend(loader.load())
            print(f"Loaded {file_type} documents")
        except Exception as e:
            print(f"Error loading {file_type} documents: {e}")
    
    return documents


def create_vector_store(documents: List, force_refresh: bool = False) -> FAISS:
    """Create or load a vector store from documents."""
    embeddings = get_embeddings_model()
    
    # Check if vector store already exists
    if os.path.exists(VECTOR_STORE_PATH) and not force_refresh:
        print("Loading existing vector store...")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
    # Create a new vector store
    print("Creating new vector store...")

    # Check if we have any documents to process
    if not documents:
        print("No documents found or loaded. Creating empty vector store with default document.")
        # Create a simple document to avoid the IndexError
        documents = [{"page_content": "This is a default document.", "metadata": {"source": "default"}}]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Double check we have splits to process
    if not splits:
        print("No text chunks created. Creating a default chunk.")
        splits = [{"page_content": "This is a default document.", "metadata": {"source": "default"}}]
        
    try:
        vector_store = FAISS.from_documents(splits, embeddings)
        
        # Save the vector store
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
        
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        # Create an empty vector store with a single document as fallback
        dummy_doc = [{"page_content": "Dummy content for initialization", "metadata": {"source": "initialization"}}]
        vector_store = FAISS.from_documents(dummy_doc, embeddings)
        return vector_store


def get_vector_store(data_dir: Optional[str] = None, force_refresh: bool = False) -> FAISS:
    """Get the vector store, creating it if necessary."""
    if force_refresh and data_dir:
        documents = load_documents(data_dir)
        return create_vector_store(documents, force_refresh=True)
    
    # Try to load existing vector store
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            print("Loading existing vector store...")
            return FAISS.load_local(VECTOR_STORE_PATH, get_embeddings_model(), allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading vector store: {e}")
    
    # If we got here, we need to create a new one
    if not data_dir:
        print("No vector store exists and no data directory provided, using default data directory")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    documents = load_documents(data_dir)
    return create_vector_store(documents)