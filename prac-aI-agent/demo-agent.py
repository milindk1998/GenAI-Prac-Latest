import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()


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
    route = llm.invoke(
        "Reply only CALCULATION_AGENT or RESEARCH_AGENT.\n"
        f"Question: {question}"
    ).content.upper()
    return "CALCULATION_AGENT" if "CALCULATION_AGENT" in route else "RESEARCH_AGENT"


def ask(agent, question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    return messages[-1].content if messages else ""


def run_multi_agent(question: str) -> dict:
    selected = route_question(question)
    agent = calc_agent if selected == "CALCULATION_AGENT" else research_agent
    answer = ask(agent, question)

    return {
        "question": question,
        "answer": answer,
        "selected_agent": selected,
    }


if __name__ == "__main__":
    question = "Add 20 and 30"
    run = run_multi_agent(question)
    print(f"Question: {question}")
    print(f"Answer: {run['answer']}")
