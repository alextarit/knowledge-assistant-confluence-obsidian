import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    # Settings for OpenAI compatible API
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_DEFAULT_MODEL: str = "gpt-4.1"
    
    TEMPERATURE: float = 0.3
    MAX_TOKENS: Optional[int] = 4096

    # LangSmith tracing
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "knowledge-assistant"
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    
    def configure_langsmith(self):
        """Configure LangSmith environment variables."""
        if self.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT
            os.environ["LANGSMITH_TRACING"] = "true" if self.LANGSMITH_TRACING else "false"
            os.environ["LANGSMITH_ENDPOINT"] = self.LANGSMITH_ENDPOINT
            os.environ["LANGSMITH_API_KEY"] = self.LANGSMITH_API_KEY
    
    # Settings for Confluence MCP Server
    CONFLUENCE_MCP_URL: str = "http://127.0.0.1:9000/mcp"
    CONFLUENCE_ACCESS_TOKEN: str = ""
    
    # Settings for Obsidian MCP Server  
    OBSIDIAN_MCP_URL: str = "http://127.0.0.1:3010/mcp"
    
    # Settings for Agent
    MAX_RECURSION_LIMIT: int = 50
    SUMMARIZATION_TRIGGER_TOKENS: int = 4096
    
    # Directory with prompts
    SYSTEM_PROMPT_DIR: str = "prompts"
    
    # Settings for Human-in-the-loop policy
    ENABLE_HUMAN_APPROVAL: bool = True
    
    @property
    def confluence_mcp_config(self) -> dict:
        """Get Confluence MCP server config."""
        return {"confluence": {"url": self.CONFLUENCE_MCP_URL, "transport": "streamable_http"}}
    
    @property
    def obsidian_mcp_config(self) -> dict:
        """Get Obsidian MCP server config."""
        return {"obsidian": {"url": self.OBSIDIAN_MCP_URL, "transport": "streamable_http"}}

settings = Settings()