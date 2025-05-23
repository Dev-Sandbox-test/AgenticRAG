import streamlit as st
import asyncio

from agents.retriever import RetrieverAgent
from agents.generator import GeneratorAgent
from utils.helpers import ensure_directories, format_agent_response, create_env_template

ensure_directories()
create_env_template()

if "logs" not in st.session_state:
    st.session_state.logs = []

st.title("Multi-Agentic RAG System")

query = st.text_input("Enter your query:", "")

if st.button("Submit"):
    st.session_state.logs.clear()
    response_placeholder = st.empty()
    with st.spinner("Running multi-agent workflow"):
        async def run_multi_agent_query():
            retriever_agent = RetrieverAgent()
            generator_agent = GeneratorAgent()

            retrieval_result =await retriever_agent.aquery(query)
            context = retrieval_result.get("output", "")
            response = await generator_agent.agenerate(f"Context from retriever: \n{context}\n\nUser question: {query}")
            return format_agent_response(response)
        
        final_response = asyncio.run(run_multi_agent_query())
    st.success("Response generated!")
    st.write("=== Generated Response ===")
    st.write(final_response)

# with st.expander("Logs", expanded=True):
#     if st.session_state.logs:
#         for log in st.session_state.logs[-20:]:
#             st.write(log)
#     else:
#         st.write("No background output yet.")