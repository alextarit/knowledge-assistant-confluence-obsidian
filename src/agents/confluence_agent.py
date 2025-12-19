import os

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain.agents.middleware import ToolRetryMiddleware

from config.settings import settings


def load_confluence_prompt() -> str:
    """Load system prompt for Confluence agent."""
    prompt_path = os.path.join(settings.SYSTEM_PROMPT_DIR, "confluence_agent_prompt.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def create_confluence_agent(llm: BaseChatModel, tools: list):
    """Create Confluence agent with provided tools."""
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=load_confluence_prompt(),
        name="confluence_agent",
        middleware=[
            ToolRetryMiddleware(
                max_retries=3,
                initial_delay=1.0,
                backoff_factor=2.0
            ),
        ],
    )
