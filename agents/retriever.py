from typing import List, Dict, Any, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.retrievers import BaseRetriever
from langchain_openai import AzureChatOpenAI

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    MAX_RELEVANT_CHUNKS
)
from vectorstore import get_vector_store


class RetrieverAgent:
    """Agent for retrieving relevant information from documents."""
    
    def __init__(self, data_dir: Optional[str] = None, force_refresh: bool = False):
        """Initialize the retriever agent."""
        self.vector_store = get_vector_store(data_dir, force_refresh)
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0
        )
        self.retriever = self._create_retriever()
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _create_retriever(self) -> BaseRetriever:
        """Create the document retriever."""
        return self.vector_store.as_retriever(
            search_kwargs={"k": MAX_RELEVANT_CHUNKS}
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create the tools for the agent."""
        retriever_tool = create_retriever_tool(
            self.retriever,
            name="document_search",
            description="Searches for information in the document collection. Use this tool when you need to find specific information from the documents."
        )
        
        return [retriever_tool]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor."""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # ...existing code...
        retriever_prompt = PromptTemplate.from_template("""
        You are an expert researcher and retriever agent. Your job is to find the most relevant information 
        from a knowledge base to answer user questions or provide context for other tasks.

        Chat History: {chat_history}

        To search for information, use the document_search tool.
        Always provide the source of your information if it came from the documents.
        If the information isn't in the documents, clearly state that you don't have that information.

        Question: {input}

        {agent_scratchpad}

        Think through the question step by step, considering:
        1. What specific information am I looking for?
        2. What search terms would best help me find this information?
        3. How should I combine and present the information I find?

        Let's begin the search process:
        """)
        # ...existing code...
        
        agent = create_openai_tools_agent(
            self.llm,
            self.tools,
            retriever_prompt
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    async def aquery(self, query: str) -> Dict[str, Any]:
        """Query the agent asynchronously."""
        return await self.agent_executor.ainvoke({"input": query})
    
    def query(self, query: str) -> Dict[str, Any]:
        """Query the agent synchronously."""
        return self.agent_executor.invoke({"input": query})