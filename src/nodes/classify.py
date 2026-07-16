"""Classify node: assign category and priority to a ticket."""
from typing import Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import CLASSIFY_MODEL
from src.prompts import load_prompts


class Classification(BaseModel):
    """Structured classifier output — unparseable text is impossible by construction."""
    category: Literal["billing", "bug", "access_request", "sales_lead", "other"]
    priority: Literal["P0", "P1", "P2", "P3"]
    reasoning: str = Field(description="One sentence explaining the choice.")


def make_classify_node(prompt_version="v1"):
    prompts = load_prompts(prompt_version)
    llm = ChatAnthropic(
        model=CLASSIFY_MODEL,     # claude-haiku-4-5
        temperature=0,            # verified supported on Haiku; required for eval reproducibility
        max_tokens=1024,          # must be explicit
    ).with_structured_output(Classification)

    def classify_node(state):
        text = state.get("normalized") or state["raw_text"]
        result = llm.invoke([
            ("system", prompts.CLASSIFY_SYSTEM),
            ("human", text),
        ])
        return {
            "category": result.category,
            "priority": result.priority,
            "trace": ["classify: {}/{} — {}".format(result.category, result.priority, result.reasoning)],
        }

    return classify_node
