# RAG-LangChain Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Chroma**, and **Hugging Face** models to provide context-aware answers based on uploaded documents. This application features a user-friendly **Streamlit** interface, allowing users to upload and query Markdown (`.md`), PDF (`.pdf`), and plain text (`.txt`) files.

---

## 🚀 Project Overview

The RAG-LangChain Chatbot enables users to upload documents, process them into a local vector store, and ask questions answered using relevant document content. The system leverages **LangChain** for orchestration, **HuggingFace** for embeddings and language models, and **Chroma** for persistent storage. The **Streamlit** interface provides an intuitive way to manage documents and interact with the chatbot.

---

## 🧰 Tech Stack

- **LangChain**: Orchestrates document processing, text splitting, and prompt management.
- **HuggingFaceEmbeddings (`all-MiniLM-L6-v2`)**: Generates semantic embeddings for document chunks.
- **ChromaDB**: Stores document embeddings in a local persistent vector store.
- **Hugging Face Endpoint (`microsoft/Phi-3-mini-4k-instruct`)**: Powers answer generation with a lightweight LLM.
- **Streamlit**: Provides a web-based interface for document uploads and chat interactions.
- **Python Libraries**: Includes `dotenv`, `tempfile`, and `logging` for environment management, file handling, and logging.

---

## ⚙️ How It Works

### 1. Document Upload and Processing

- Users upload `.md`, `.pdf`, or `.txt` files via the Streamlit interface.
- Files are temporarily saved and loaded using:
  - `DirectoryLoader`
  - `PyPDFLoader`
  - `TextLoader`
- Documents are split into 1000-character chunks with 100-character overlaps using `RecursiveCharacterTextSplitter`.

### 2. Vector Embedding

- Each chunk is embedded using the `all-MiniLM-L6-v2` model from HuggingFace.
- Embeddings are stored in a local Chroma vector store (rebuilt from scratch per session).

### 3. Query Processing

- Users enter questions in the Streamlit UI.
- A similarity search retrieves the top 3 most relevant chunks.
- These chunks are formatted into a prompt using `ChatPromptTemplate`.

### 4. Answer Generation

- The prompt (including context + question) is sent to the `microsoft/Phi-3-mini-4k-instruct` model via Hugging Face endpoint.
- The response is generated and displayed in the chat, along with source references.

### 5. User Interface

- Responsive **Streamlit UI** with:
  - Sidebar for document uploads and About section.
  - Main area with a scrollable chat interface.
  - Visual feedback (e.g., spinners, balloons, success messages).

---

## ✅ Features

- **Multi-format Support**: Supports `.md`, `.pdf`, and `.txt`.
- **Interactive UI**: Chat-style interface with document management.
- **Context-Aware Answers**: RAG ensures grounded, document-based responses.
- **Persistent Storage**: ChromaDB enables fast, local retrieval.
- **Error Handling**: Clear feedback for invalid inputs or processing failures.

---

## 🛠️ Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-langchain-chatbot
```

### 2. Install Dependecies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
HUGGINGFACEHUB_API_TOKEN=your-api-key
```

### 4. Run the Application
```bash
streamlit run main.py
```
Visit http://localhost:8501 in your browser.
