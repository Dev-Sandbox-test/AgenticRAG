from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT
)
from agents.retriever import RetrieverAgent


class GeneratorAgent:
    """Agent for generating content using retrieved information."""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0.7  # Slightly more creative
        )
        self.agent_executor = self._create_agent_executor()
    
    def _create_agent_executor(self) -> AgentExecutor:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        generator_prompt = PromptTemplate.from_template("""
        You are an expert content generation assistant.
        Use the provided context to answer the user's question as accurately as possible.

        Chat History: {chat_history}

        Context: {input}

        {agent_scratchpad}

        Generate a comprehensive and accurate response, citing sources if available.
        """)
        
        agent = create_openai_tools_agent(
            self.llm,
            [], # No tools for this agent
            generator_prompt
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=[],
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    async def agenerate(self, context: str) -> Dict[str, Any]:
        """Generate content asynchronously."""
        return await self.agent_executor.ainvoke({"input": context})
    
    def generate(self, context: str) -> Dict[str, Any]:
        """Generate content synchronously."""
        return self.agent_executor.invoke({"input": context})