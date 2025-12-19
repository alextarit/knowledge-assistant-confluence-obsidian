import asyncio
import logging
import uuid
from typing import Optional

from agents.supervisor_graph import SupervisorSystem
from config.settings import settings
from logger.logger import init_logs

logger = logging.getLogger(__name__)


class KnowledgeAssistant:
    def __init__(self):
        self.system = None
        self._initialized = False
    
    async def initialize(self):
        if self._initialized:
            return
        
        logger.info("=" * 60)
        logger.info("Starting Knowledge Assistant")
        logger.info("=" * 60)
        logger.info("Model: %s", settings.OPENAI_DEFAULT_MODEL)
        logger.info("Confluence MCP: %s", settings.CONFLUENCE_MCP_URL)
        logger.info("Obsidian MCP: %s", settings.OBSIDIAN_MCP_URL)
        logger.info("Human-in-the-loop: %s", "Enabled" if settings.ENABLE_HUMAN_APPROVAL else "Disabled")
        
        self.system = SupervisorSystem()
        await self.system.initialize()
        
        self._initialized = True
        logger.info("System ready!")
    
    async def chat(self, user_input: str, thread_id: Optional[str] = None) -> tuple[dict, str]:
        await self.initialize()
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            logger.info("New dialog: %s...", thread_id[:8])
        
        try:
            response = await self.system.run(user_input, thread_id)
            return response, thread_id
        except Exception as e:
            logger.error("Error processing request: %s", e, exc_info=True)
            return {"status": "error", "content": str(e)}, thread_id
    
    async def interactive_session(self):
        await self.initialize()
        
        logger.info("=" * 60)
        logger.info("Interactive mode")
        logger.info("=" * 60)
        logger.info("Commands: 'quit'/'exit' - exit, 'new' - new dialog")
        logger.info("=" * 60)
        
        thread_id = None
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit"]:
                    logger.info("Goodbye!")
                    break
                
                if user_input.lower() == "new":
                    thread_id = None
                    logger.info("New dialog started")
                    continue
                
                result, thread_id = await self.chat(user_input, thread_id)
                content = result.get("content", result) if isinstance(result, dict) else result
                print(f"\nAssistant: {content}")
                
            except KeyboardInterrupt:
                logger.warning("Interrupted. Type 'quit' to exit.")
            except Exception as e:
                logger.error("Error: %s", e, exc_info=True)


async def main():
    init_logs()
    settings.configure_langsmith()
    assistant = KnowledgeAssistant()
    
    try:
        await assistant.interactive_session()
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.critical("Critical error: %s", e, exc_info=True)
        return 1
    
    return 0


def run():
    return asyncio.run(main())


if __name__ == "__main__":
    exit(run())
