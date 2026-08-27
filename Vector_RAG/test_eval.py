from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rag import from_rag
import os

load_dotenv()


# Wrap ChatGroq so deepeval can use it as an LLM judge
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(api_key=os.environ.get("api_key"), model=os.environ.get("modelAsJudge"), temperature=0.0)

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


def test_evaluate_rag():

    answer, question, context = from_rag("What is the FEATURES OF CORPORATE POLICY")

    test_case = LLMTestCase(input=question, actual_output=answer.content, retrieval_context=context)

    metrics = [AnswerRelevancyMetric(model=GroqJudge(), threshold=0.8, async_mode=False),
               ContextualRelevancyMetric(model=GroqJudge(), threshold=0.7, async_mode=False)]

    #  Deepeval registration is not used because test_eval.py:41 is commented 
    #  and you only do manual metric.measure loops.
    #  So Deepeval reports: No test cases found, please try again.

    # results = evaluate(test_cases=[test_case], metrics=metrics)

    # Quality gate evaluation
    failed = False

    for metric in metrics:
        metric.measure(test_case)

        if metric.error:
            print(f"Error: {metric.error}")
            failed = True

        elif metric.score is None:
            print("Error: Score is None")
            failed = True

        elif metric.score < metric.threshold:
            print(f"Quality gate FAILED — {metric.__class__.__name__}: {metric.score}")
            failed = True
        else:
            print(f"Quality gate PASSED — {metric.__class__.__name__}: {metric.score}")

        if failed:
            raise SystemExit("Evaluation ERRORED — Check the logs for details.")


if __name__ == "__main__":
    test_evaluate_rag()

