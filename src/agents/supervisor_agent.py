import logging
import os

from langchain.tools import tool, ToolRuntime

from config.settings import settings

logger = logging.getLogger(__name__)


def load_supervisor_prompt() -> str:
    """Load system prompt for Supervisor agent."""
    prompt_path = os.path.join(settings.SYSTEM_PROMPT_DIR, "supervisor_agent_prompt.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def create_supervisor_tools(confluence_agent, obsidian_agent) -> list:
    """Create tools that wrap sub-agents for Supervisor."""
    
    @tool
    async def search_confluence(request: str, runtime: ToolRuntime) -> str:
        """Search in Confluence documentation."""
        original_user_message = next(
            (msg for msg in runtime.state["messages"] if msg.type == "human"),
            None
        )
        
        if original_user_message:
            prompt = (
                f"User's original request:\n{original_user_message.content}\n\n"
                f"Your task:\n{request}"
            )
        else:
            prompt = request
        
        logger.debug("Confluence agent request: %s", request)
        
        result = await confluence_agent.ainvoke({
            "messages": [{"role": "user", "content": prompt}]
        })
        
        return result["messages"][-1].content
    
    @tool
    async def manage_obsidian_notes(request: str, runtime: ToolRuntime) -> str:
        """Manage personal notes in Obsidian vault."""
        original_user_message = next(
            (msg for msg in runtime.state["messages"] if msg.type == "human"),
            None
        )
        
        if original_user_message:
            prompt = (
                f"User's original request:\n{original_user_message.content}\n\n"
                f"Your task:\n{request}"
            )
        else:
            prompt = request
        
        logger.debug("Obsidian agent request: %s", request)
        
        result = await obsidian_agent.ainvoke({
            "messages": [{"role": "user", "content": prompt}]
        })
        
        return result["messages"][-1].content
    
    return [search_confluence, manage_obsidian_notes]
