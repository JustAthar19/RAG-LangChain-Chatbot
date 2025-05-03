import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_huggingface import  ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

llm = HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        temperature=0.7,
        max_new_tokens=512,
)

chat_model = ChatHuggingFace(llm=llm)



def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The Query Text")
    # args = parser.parse_args()
    # query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    while True:
        print("\n\n------------------------")
        question = input("Input your question: ")
        print("\n\n")
        if question == 'q':
            break
        
        results = db.similarity_search_with_relevance_scores(question, k=3) # k=3, the number of return we want to retrieve
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _scores in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)
    
        response = chat_model.invoke(prompt)  
        sources = [doc.metadata.get("source", None) for doc, _score in results]
    
        formatted_response = f"Response: {response.content}\nSources: {sources}"
        print(formatted_response)

if __name__ == '__main__':
    main()