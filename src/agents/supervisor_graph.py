import logging

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolRetryMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agents.confluence_agent import create_confluence_agent
from agents.obsidian_agent import create_obsidian_agent
from agents.supervisor_agent import create_supervisor_tools, load_supervisor_prompt
from config.settings import settings
from utils.llm_retry import create_llm

logger = logging.getLogger(__name__)


class SupervisorSystem:
    """Multi-agent system coordinator."""
    
    def __init__(self):
        self.checkpointer = None
        self._initialized = False
        self.llm = None
        self.confluence_mcp = None
        self.obsidian_mcp = None
        self._current_graph = None
    
    async def initialize(self):
        """Initialize system components."""
        if self._initialized:
            return
        
        logger.info("Initializing Supervisor system...")
        
        self.llm = create_llm()
        self.checkpointer = MemorySaver()
        
        self.confluence_mcp = MultiServerMCPClient(settings.confluence_mcp_config)
        self.obsidian_mcp = MultiServerMCPClient(settings.obsidian_mcp_config)
        
        logger.info("LLM and MCP clients initialized")
        self._initialized = True
    
    async def _ensure_graph(self):
        """Create or return existing graph."""
        if self._current_graph is None:
            confluence_tools = await self.confluence_mcp.get_tools()
            obsidian_tools = await self.obsidian_mcp.get_tools()
            
            logger.debug("Confluence tools: %d, Obsidian tools: %d", 
                        len(confluence_tools), len(obsidian_tools))
            
            confluence_agent = create_confluence_agent(self.llm, confluence_tools)
            obsidian_agent = create_obsidian_agent(self.llm, obsidian_tools)
            
            supervisor_tools = create_supervisor_tools(confluence_agent, obsidian_agent)
            system_prompt = load_supervisor_prompt()
            
            interrupt_config = ["tools"] if settings.ENABLE_HUMAN_APPROVAL else None
            
            self._current_graph = create_agent(
                model=self.llm,
                tools=supervisor_tools,
                system_prompt=system_prompt,
                checkpointer=self.checkpointer,
                middleware=[
                    SummarizationMiddleware(
                        model=self.llm,
                        trigger=("tokens", settings.SUMMARIZATION_TRIGGER_TOKENS),
                        keep=("messages", 10),
                    ),
                    ToolRetryMiddleware(
                        max_retries=3,
                        initial_delay=1.0,
                        backoff_factor=2.0
                    ),
                ],
                interrupt_before=interrupt_config,
                name="supervisor",
            )
        return self._current_graph
    
    async def run(self, user_input: str, thread_id: str):
        """Run the multi-agent system."""
        await self.initialize()
        
        graph = await self._ensure_graph()
        
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.MAX_RECURSION_LIMIT,
        }
        
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        return self._process_result(result)
    
    async def resume_after_approval(self, thread_id: str, approved: bool = True):
        """Resume execution after human approval."""
        await self.initialize()
        
        graph = await self._ensure_graph()
        
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.MAX_RECURSION_LIMIT,
        }
        
        if approved:
            result = await graph.ainvoke(Command(resume=True), config=config)
        else:
            result = await graph.ainvoke(
                Command(resume={"action": "rejected"}),
                config=config
            )
        
        return self._process_result(result)
    
    def _process_result(self, result):
        """Process graph result."""
        messages = result.get("messages", [])
        
        if not messages:
            return {"status": "error", "content": "No messages in result"}
        
        last_message = messages[-1]
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return {
                "status": "pending_approval",
                "tool_calls": [
                    {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                    for tc in last_message.tool_calls
                ],
                "content": "Actions require approval",
            }
        
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return {"status": "complete", "content": message.content}
        
        return {"status": "error", "content": "Could not get response"}
