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


def make_tool_call(name: str, params: dict) -> ToolCall:
    return ToolCall(name=name, input_parameters=params)


def test_multi_agent_deepeval():
    test_cases = [
        LLMTestCase(
            input="Add 20 and 30",
            actual_output="50",
            tools_called=[make_tool_call("calculate", {"a": 20, "b": 30})],
            expected_tools=[make_tool_call("calculate", {"a": 20, "b": 30})],
        ),
        LLMTestCase(
            input="Search latest LangChain updates",
            actual_output="Search results for 'latest LangChain updates'",
            tools_called=[make_tool_call("search", {"query": "latest LangChain updates"})],
            expected_tools=[make_tool_call("search", {"query": "latest LangChain updates"})],
        ),
    ]

    metrics=[
                ToolCorrectnessMetric(model=GroqJudge(), threshold=1.0, async_mode=False),
                ArgumentCorrectnessMetric(model=GroqJudge(), threshold=1.0, async_mode=False),
            ]

    evaluate(test_cases=test_cases, metrics=metrics)

    # failed = False

    # for metric in metrics:
    #     for test_case in test_cases:
    #         metric.measure(test_case)
    #         if metric.error:
    #             print(f"Error: {metric.error}")
    #             failed = True
    #         elif metric.score is None:
    #             print("Error: Score is None")
    #             failed = True
    #         elif metric.score < metric.threshold:
    #             print(f"Quality gate FAILED — {metric.__class__.__name__}: {metric.score}")
    #             failed = True
    #         else:
    #             print(f"Quality gate PASSED — {metric.__class__.__name__}: {metric.score}")

    # if failed:
    #     raise SystemExit("Evaluation ERRORED — Check the logs for details.")


if __name__ == "__main__":
    test_multi_agent_deepeval()
