# RAG langchain chatbot 

LangChain RAG Chatbot is a local question-answering sA simple Retrieval-Augmented Generation (RAG) chatbot that uses **LangChain**, **Chroma**, and **Hugging Face models** to answer questions based on custom Markdown documents.

---

# Project Overview
This chatbot loads `.md` files from a local directory, splits the content into manageable chunks, embeds them into a vector store using **MiniLM**, and serves context-aware answers using **Phi-3 Mini** LLM from Hugging Face. It's designed to run locally with a CLI interface.

---

## Tech Stack

- **LangChain** – for orchestration, text splitting, prompting
- **HuggingFaceEmbeddings** – (`all-MiniLM-L6-v2`) for semantic vectorization
- **ChromaDB** – local persistent vector store
- **Hugging Face Endpoint** – (`microsoft/Phi-3-mini-4k-instruct`) for generating responses


---

##  How It Works

1. **Document Loading**  
   Markdown files from `data/books/` are loaded and split into chunks.

2. **Vector Embedding**  
   Each chunk is converted to an embedding using MiniLM and stored in Chroma.

3. **Similarity Search + Prompting**  
   At runtime, user questions are matched with top-k relevant chunks, which are passed to an LLM via prompt template.

4. **Answer Generation**  
   A Hugging Face-hosted Phi-3 Mini model returns an answer grounded in the provided context.

---

