import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate   
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    ToolCorrectnessMetric, 
    ArgumentCorrectnessMetric,
)

from agent import boss_agent_with_trace

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(api_key=os.environ.get("api_key"), 
                              model=os.environ.get("modelAsJudge"), temperature=0.0)

    def load_model(self):
        return self.model

    def get_model_name(self):
        return os.environ.get("modelAsJudge")

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content


def _to_tool_call(name: str, params: dict):
    """Support small schema differences across DeepEval versions."""
    try:
        return ToolCall(name=name, input_parameters=params)
    except Exception:
        try:
            return ToolCall(name=name, arguments=params)
        except Exception:
            return ToolCall(tool_name=name, input_parameters=params)


def test_evaluate_multi_agent():
    run = boss_agent_with_trace("Add 20 and 30")
    question = run["question"]
    answer = run["answer"]
    selected_agent = run["selected_agent"]
    runtime_tool_calls = run["runtime_tool_calls"]

    # Easy interview explanation: these asserts prove real runtime behavior.
    assert selected_agent == "CALCULATION_AGENT", (
        f"Expected CALCULATION_AGENT, got {selected_agent}. route_raw={run['route_raw']}"
    )

    calc_calls = [call for call in runtime_tool_calls if call.get("name") == "calculate"]
    assert calc_calls, f"No runtime calculate call found. runtime_tool_calls={runtime_tool_calls}"

    args = calc_calls[0].get("input_parameters", {})
    a = int(args.get("a"))
    b = int(args.get("b"))
    assert {a, b} == {20, 30}, f"Expected calculate args a=20 and b=30, got args={args}"

    tools_called = [
        _to_tool_call(call["name"], call.get("input_parameters", {}))
        for call in runtime_tool_calls
    ]

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        tools_called=tools_called,
        expected_tools=[_to_tool_call("calculate", {"a": 20, "b": 30})],
    )

    metrics = [
        ToolCorrectnessMetric(model=GroqJudge(), threshold=0.8),
        ArgumentCorrectnessMetric(model=GroqJudge(), threshold=0.8),
    ]

    evaluate(test_cases=[test_case], metrics=metrics)

if __name__ == "__main__":
    test_evaluate_multi_agent()

