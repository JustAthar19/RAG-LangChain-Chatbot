from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/books/"

def load_document():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def main():
    documents = load_document()
    chunks = split_text(documents)
    save_to_chroma(chunks)    


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # create a new db from the documents
    db = Chroma.from_documents(
        chunks, 
        HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
        persist_directory=CHROMA_PATH
    )
    
    print(f"saved {len(chunks)} to {CHROMA_PATH}")

if __name__ == "__main__":
    main()