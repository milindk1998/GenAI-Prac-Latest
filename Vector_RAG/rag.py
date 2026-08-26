from dotenv import load_dotenv
load_dotenv()
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

def from_rag(query):
    # load pdf
    document_load = PyPDFLoader("Policy.pdf").load()

    # chunk the document
    chunk_doc = RecursiveCharacterTextSplitter(separators=['/n/n', '/n', ' ', ''],chunk_size=1000, chunk_overlap=200).split_documents(document_load)

    # embed the document
    embeddings = HuggingFaceEmbeddings(model=os.environ.get("embedModel"))

    # store in vector db
    vectorstore = Chroma.from_documents(documents=chunk_doc, embedding=embeddings, persist_directory="./chromadb")

    # retrieve relevent documents
    rel_doc = vectorstore.similarity_search(query, k=2)

    # context for the model
    context_list = [doc.page_content for doc in rel_doc]
    context = "\n\n".join(context_list)

    prompt = f"""
    You are a helpful assistant that answers questions based on the context below. 
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the
    provided context: {context}
    question: {query}    
    """

    # load model
    llm = ChatGroq(model=os.environ.get("testmodel"), api_key=os.environ.get("api_key"))

    # get answer
    answer = llm.invoke(prompt)

    return answer, query, context_list

if __name__ == "__main__":
    query = "What is the FEATURES OF CORPORATE POLICY"
    answer, query = from_rag(query)
    print(f"Query: {query}\nAnswer: {answer.content}")

