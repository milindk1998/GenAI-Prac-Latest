from dotenv import load_dotenv
import os
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(api_key=os.environ.get("api_key"),
               model=os.environ.get("testmodel"))

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant.")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({'messages': [HumanMessage(content="Hello, how are you?")]})

print(response)