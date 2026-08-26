import asyncio
import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp.client import Client

from server import mcp

load_dotenv()

llm = ChatGroq(
    model=os.environ.get("testmodel"),
    api_key=os.environ.get("api_key"),
    temperature=0.0,
)


def route_question(question: str) -> str:
    route = llm.invoke(
        "Reply only ADD_AGENT or SUBTRACT_AGENT or MULTIPLY_AGENT.\n"
        f"Question: {question}"
    ).content.upper()
    if "SUBTRACT_AGENT" in route:
        return "SUBTRACT_AGENT"
    if "MULTIPLY_AGENT" in route:
        return "MULTIPLY_AGENT"
    return "ADD_AGENT"


async def run_client(question: str):
    route = route_question(question)
    nums = re.findall(r"-?\d+(?:\.\d+)?", question)
    if len(nums) < 2:
        raise ValueError("Need two numbers in question")
    a, b = map(float, nums[:2])
    tool_name = {
        "ADD_AGENT": "add",
        "SUBTRACT_AGENT": "subtract",
        "MULTIPLY_AGENT": "multiply",
    }[route]

    async with Client(mcp) as client:
        result = await client.call_tool(tool_name, {"a": a, "b": b})
        answer = result.content[0].text

    return answer


async def main():
    questions = ["Add 20 and 30", "Subtract 50 and 10", "Multiply 7 and 8"]
    for q in questions:
        print(await run_client(q))


if __name__ == "__main__":
    asyncio.run(main())

