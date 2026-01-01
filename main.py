import streamlit as st
import os
import shutil
import tempfile
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv

os.getenv("")


import logging #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "messages" not in st.session_state:
    st.session_state.messages = []

def load_document(data_path):
    documents = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if file.endswith(".md"):
            loader = DirectoryLoader(data_path, glob="*.md")
            documents.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.append(loader.load())
    return documents if documents else None, "No supported files found." if not documents else None

def split_text(documents):
    logger.info("Splitting Documents into chunks")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks, None
    except Exception as e:
        return None, f"Error splitting documents: {str(e)}"


def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"}
    )

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


# function to create the databaa
def save_to_chroma(new_chunks):
    logger.info("Saving Chunks to Chroma Dataset")
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=get_embedding_function()
        )
        # Calculate Page IDs.
        chunks_with_ids = calculate_chunks_id(new_chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add")
        return db, None
    except Exception as e:
        return None, f"Error saving to Chroma: {str(e)}"


def calculate_chunks_id(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}/{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}/{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata['id'] = chunk_id
    
    return chunks



def process_documents(uploaded_files):
    if not uploaded_files:
        return "Please upload at least one markdown file."
    
    with st.spinner("Processing uploaded documents..."):
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
            
            # Load and process documents
            documents, error = load_document(temp_dir)
            if error:
                return error
            
            chunks, error = split_text(documents)
            if error:
                return error
            
            db, error = save_to_chroma(chunks)
            if error:
                return error
            
            st.session_state.db = db
            return f"Successfully processed {len(chunks)} document chunks."



llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.7, max_new_tokens=512)
chat_model = ChatHuggingFace(llm=llm)

def query_database(question):
    if not st.session_state.db:
        return "No documents loaded. Please upload documents first.", None
    
    with st.spinner("Generating response..."):
        try:
            results = st.session_state.db.similarity_search_with_score(question, k=5)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=question)
            response = chat_model.invoke(prompt)
            sources = [doc.metadata.get("source", None) for doc, _score in results]
            return response.content, sources
        except Exception as e:
            return f"Error generating response: {str(e)}", None

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG-LangChain Chatbot")

    

with st.sidebar:
    with st.expander("📤 Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload .md, .pdf, or .txt files",
                type=["md", "pdf", "txt"],
                accept_multiple_files=True,
                help="Select one or more files to process"
            )
            
        if st.button("Process Documents", disabled=st.session_state.processing):
            if uploaded_files:
                st.session_state.processing = True
                result = process_documents(uploaded_files)
                st.session_state.processing = False
                if "Error" in result or "Please" in result:
                    st.error(result)
                else:
                    st.success(result)
            else:
                st.warning("Please upload at least one file.")
    
    


# Main content
chat_container = st.container()

# Input for user question
with st.form(key="query_form"):
    question = st.text_input("Ask anything about your document")
    submit_button = st.form_submit_button("Submit")

# Handle query submission
if submit_button and question:
    st.session_state.messages.append({"role": "user", "content": question})
    response, sources = query_database(question)
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])