from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("api_key")
model = os.environ.get("testmodel")

def llm(question):
    llm = ChatGroq(api_key=api_key, model=model)

    messages = [SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=question)]

    response = llm.invoke(messages)

    return response

if __name__ == "__main__":
    question = "Hello, how are you?"
    response = llm(question)
    print(response.content)
    print("Model Name is ", response.response_metadata["model_name"])
    metadata = response.response_metadata
    for key, value in metadata.items():
        print(f"{key}: {value}")