from langchain_groq import ChatGroq
import os, json
from dotenv import load_dotenv
load_dotenv()

MEMORY_FILE = "mem.json"
SYSTEM_PROMPT = "You are a helpful assistant. If you don't know the answer, say 'I don't know.'"

llm = ChatGroq(api_key=os.environ.get("api_key"), model=os.environ.get("modelAsJudge"), temperature=0.0)

def chatbot(question):

    #load the mem
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            data=json.load(f)

    else:
        data=[{
            "role":"system",
            "content": SYSTEM_PROMPT
        }]

    data.append({
        "role": "user",
        "content": question
    })

    response = llm.invoke(data)

    data.append({
        "role":"assistant",
        "content": response.content
    })

    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    return response.content

if __name__ == "__main__":
    while True:
        question = str(input("You: "))
        if question.lower() in ['exit', 'quit']:
            break
        answer = chatbot(question)
        print("Response", answer)
