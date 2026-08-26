import asyncio
import os
import re

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import MCPUseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from langchain_groq import ChatGroq
from mcp.client import Client

from client import route_question
from server import mcp

load_dotenv()


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            api_key=os.environ.get("api_key"),
            model=os.environ.get("modelAsJudge"),
            temperature=0.0,
        )

    def load_model(self):
        return self.model

    def get_model_name(self):
        return os.environ.get("modelAsJudge") or "groq-judge"

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.model.ainvoke(prompt)).content

MCP_SERVER_DEF = MCPServer(
    server_name="Calculator",
    transport="stdio",
    available_tools=[
        {"name": "add"},
        {"name": "subtract"},
        {"name": "multiply"},
    ],
)


async def _build_case(question: str) -> LLMTestCase:
    nums = re.findall(r"-?\d+(?:\.\d+)?", question)
    if len(nums) < 2:
        raise ValueError("Need two numbers in question")
    a, b = map(float, nums[:2])

    tool_name = {
        "ADD_AGENT": "add",
        "SUBTRACT_AGENT": "subtract",
        "MULTIPLY_AGENT": "multiply",
    }[route_question(question)]

    async with Client(mcp) as client:
        result = await client.call_tool(tool_name, {"a": a, "b": b})

    text = next(
        (str(block.text) for block in getattr(result, "content", []) if getattr(block, "text", None)),
        str(getattr(result, "structured_content", "")),
    )

    return LLMTestCase(
        input=question,
        actual_output=text,
        mcp_servers=[MCP_SERVER_DEF],
        mcp_tools_called=[MCPToolCall(name=tool_name, args={"a": a, "b": b}, result=result)],
    )


def test_mcp_client_use_metric():
    questions = ["Add 20 and 30", "Subtract 50 and 10", "Multiply 7 and 8"]
    cases = [asyncio.run(_build_case(q)) for q in questions]
    evaluate(test_cases=cases, metrics=[MCPUseMetric(model=GroqJudge(), threshold=0.8)])


if __name__ == "__main__":
    test_mcp_client_use_metric()

