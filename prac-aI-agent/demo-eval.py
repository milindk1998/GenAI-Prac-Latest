import os

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric, ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall
from langchain_groq import ChatGroq

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


def _tool_call(name: str, params: dict):
    # Keep compatibility across DeepEval versions.
    try:
        return ToolCall(name=name, input_parameters=params)
    except Exception:
        return ToolCall(name=name, arguments=params)


def test_multi_agent_hardcoded_with_deepeval():
    judge = GroqJudge()

    test_cases = [
        LLMTestCase(
            input="Add 20 and 30",
            actual_output="50",
            tools_called=[_tool_call("calculate", {"a": 20, "b": 30})],
            expected_tools=[_tool_call("calculate", {"a": 20, "b": 30})],
        ),
        LLMTestCase(
            input="Search latest LangChain updates",
            actual_output="Search results for 'latest LangChain updates'",
            tools_called=[_tool_call("search", {"query": "latest LangChain updates"})],
            expected_tools=[_tool_call("search", {"query": "latest LangChain updates"})],
        ),
        LLMTestCase(
            input="Add 20 and 30",
            actual_output="50",
            tools_called=[_tool_call("calculate", {"a": 20, "b": 30})],
            # Intentional mismatch for demo: expected tool is wrong on purpose.
            expected_tools=[_tool_call("search", {"query": "add 20 and 30"})],
        ),
    ]

    evaluate(
        test_cases=test_cases,
        metrics=[
            ToolCorrectnessMetric(model=judge, threshold=1.0),
            ArgumentCorrectnessMetric(model=judge, threshold=1.0),
        ],
    )


if __name__ == "__main__":
    test_multi_agent_hardcoded_with_deepeval()
