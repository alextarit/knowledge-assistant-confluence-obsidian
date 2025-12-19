import asyncio
import uuid

import streamlit as st

from agents.supervisor_graph import SupervisorSystem
from config.settings import settings

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

settings.configure_langsmith()


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "system" not in st.session_state:
        st.session_state.system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None


async def initialize_system():
    if st.session_state.system is None:
        st.session_state.system = SupervisorSystem()
    if not st.session_state.initialized:
        await st.session_state.system.initialize()
        st.session_state.initialized = True


async def process_message(user_input: str) -> dict:
    await initialize_system()
    result = await st.session_state.system.run(
        user_input,
        st.session_state.thread_id
    )
    return result


async def resume_execution(approved: bool) -> dict:
    await initialize_system()
    result = await st.session_state.system.resume_after_approval(
        st.session_state.thread_id,
        approved
    )
    return result


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("MCP Servers")
        st.info(f"**Confluence:** `{settings.CONFLUENCE_MCP_URL}`")
        st.info(f"**Obsidian:** `{settings.OBSIDIAN_MCP_URL}`")
        
        st.subheader("Model")
        st.write(f"**{settings.OPENAI_DEFAULT_MODEL}**")
        st.write(f"Temperature: {settings.TEMPERATURE}")
        
        st.subheader("Human-in-the-Loop")
        if settings.ENABLE_HUMAN_APPROVAL:
            st.success("✅ Enabled")
        else:
            st.warning("⚠️ Disabled")
        
        st.divider()
        
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.pending_approval = None
            st.session_state.system = None
            st.session_state.initialized = False
            st.rerun()
        
        st.divider()
        
        st.subheader("Capabilities")
        st.markdown("""
        - 📚 **Confluence** — search docs
        - 📝 **Obsidian** — manage notes
        """)


def render_approval_ui():
    """Render human-in-the-loop approval UI."""
    if st.session_state.pending_approval is None:
        return
    
    tool_calls = st.session_state.pending_approval
    
    st.warning("🛡️ **Actions require your approval**")
    
    for tc in tool_calls:
        with st.expander(f"🔧 Tool: `{tc['name']}`", expanded=True):
            st.json(tc["args"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            with st.spinner("Executing..."):
                try:
                    result = asyncio.run(resume_execution(approved=True))
                    st.session_state.pending_approval = None
                    
                    if result["status"] == "complete":
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["content"]
                        })
                    elif result["status"] == "pending_approval":
                        st.session_state.pending_approval = result["tool_calls"]
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.button("❌ Reject", use_container_width=True):
            st.session_state.pending_approval = None
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ Action was rejected by user."
            })
            st.rerun()


def render_chat():
    st.title("🧠 Knowledge Assistant")
    st.caption("Multi-agent system for Confluence & Obsidian")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.pending_approval:
        render_approval_ui()
        return
    
    if prompt := st.chat_input("Ask about Confluence or Obsidian..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    result = asyncio.run(process_message(prompt))
                    
                    if result["status"] == "complete":
                        st.markdown(result["content"])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["content"]
                        })
                    elif result["status"] == "pending_approval":
                        st.session_state.pending_approval = result["tool_calls"]
                        st.rerun()
                    else:
                        st.error(result["content"])
                        
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
