import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

MATH_KEYWORDS = ("add", "sum", "plus", "calculate", "total")


@tool
def calculate(a: int, b: int) -> str:
    """Return sum of two integers."""
    return str(a + b)


@tool
def search(query: str) -> str:
    """Mock search tool."""
    return f"Search results for '{query}'"


llm = ChatGroq(
    model=os.environ.get("testmodel"),
    api_key=os.environ.get("api_key"),
    temperature=0.0,
)

calc_agent = create_agent(model=llm, tools=[calculate], system_prompt="You are a calculator.")
research_agent = create_agent(model=llm, tools=[search], system_prompt="You are a researcher.")


def route_question(question: str) -> str:
    for word in MATH_KEYWORDS:
        if word in question.lower():
            return "CALCULATION_AGENT"
    return "RESEARCH_AGENT"


def run_multi_agent(question: str) -> dict:
    agent = calc_agent if route_question(question) == "CALCULATION_AGENT" else research_agent
    answer = agent.invoke({"messages": [{"role": "user", "content": question}]})["messages"][-1].content

    return question, answer


if __name__ == "__main__":
    demo_question = "Add 20 and 30 and then result add by 10"
    question, answer = run_multi_agent(demo_question)
    print("Question:", question)
    print("Answer:", answer)
