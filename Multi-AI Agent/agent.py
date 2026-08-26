import os
import json
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent



@tool
def calculate(a: int, b: int) -> str:
    """Calculate the sum of two numbers."""
    return str(a + b)

@tool
def search(query: str) -> str:
    """Search for information on the web."""
    return f"Search results for '{query}'"


llm = ChatGroq(model=os.environ.get("testmodel"), api_key=os.environ.get("api_key"), temperature=0.2)


research_agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="You are a research assistant. Use available tools to answer clearly.",
)

calculation_agent = create_agent(
    model=llm,
    tools=[calculate],
    system_prompt="You are a calculator. Use tools and show concise step-by-step reasoning.",
)


def ask(agent, question: str) -> str:
    trace = ask_with_trace(agent, question)
    return trace["answer"]


def _extract_runtime_tool_calls(messages) -> list[dict]:
    runtime_tool_calls = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tool_call in tool_calls:
            name = tool_call.get("name")
            if not name:
                continue

            args = tool_call.get("args", tool_call.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}

            if not isinstance(args, dict):
                args = {"value": args}

            runtime_tool_calls.append({
                "name": name,
                "input_parameters": args,
            })
    return runtime_tool_calls


def ask_with_trace(agent, question: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else ""
    return {
        "answer": answer,
        "runtime_tool_calls": _extract_runtime_tool_calls(messages),
    }

def boss_agent(question: str):
    run = boss_agent_with_trace(question)
    return run["question"], run["answer"]


def boss_agent_with_trace(question: str) -> dict:
    route_prompt = (
        "Classify the user question. Reply with only CALCULATION_AGENT or RESEARCH_AGENT.\n"
        f"Question: {question}"
    )
    route = llm.invoke(route_prompt).content.upper().strip()
    selected_agent = "CALCULATION_AGENT" if "CALCULATION_AGENT" in route else "RESEARCH_AGENT"
    agent = calculation_agent if selected_agent == "CALCULATION_AGENT" else research_agent
    trace = ask_with_trace(agent, question)

    return {
        "question": question,
        "answer": trace["answer"],
        "selected_agent": selected_agent,
        "runtime_tool_calls": trace["runtime_tool_calls"],
        "route_raw": route,
    }

if __name__ == "__main__":
    question = "Add 20 and 30 and result is again added by 10 then search for langchain"
    run = boss_agent_with_trace(question)
    print(f"Question: {run['question']}")
    print(f"Selected agent: {run['selected_agent']}")
    print(f"Tool calls: {run['runtime_tool_calls']}")
    print(f"Answer: {run['answer']}")
    
