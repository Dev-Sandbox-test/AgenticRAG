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
    
    def __init__(self, retriever_agent: Optional[RetrieverAgent] = None, data_dir: Optional[str] = None):
        """Initialize the generator agent."""
        self.retriever_agent = retriever_agent or RetrieverAgent(data_dir)
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0.7  # Slightly more creative
        )
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _create_tools(self) -> List[Tool]:
        """Create the tools for the generator agent."""
        retrieval_tool = Tool(
            name="retrieve_information",
            func=self.retriever_agent.query,
            description="Retrieves information from documents based on the query. Use this to gather facts and context."
        )
        
        return [retrieval_tool]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor."""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        generator_prompt = PromptTemplate.from_template("""
        You are an expert content generation assistant with access to a document retrieval system.
        Your goal is to generate high-quality, accurate content based on the user's request and 
        information from the knowledge base.
        
        Chat History: {chat_history}
        {agent_scratchpad}
        When generating content:
        1. First use the retrieve_information tool to gather relevant information.
        2. Based on the retrieved information, generate comprehensive and accurate content.
        3. Always cite your sources if the information came from specific documents.
        4. If the knowledge base doesn't contain relevant information, state that and provide 
           the best response you can based on your general knowledge.
        5. Format your response appropriately based on the user's request (e.g., bullet points, 
           paragraphs, tables, etc.).
        
        User Request: {input}
        
        Let's work through this step by step:
        """)
        
        agent = create_openai_tools_agent(
            self.llm,
            self.tools,
            generator_prompt
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    async def agenerate(self, query: str) -> Dict[str, Any]:
        """Generate content asynchronously."""
        return await self.agent_executor.ainvoke({"input": query})
    
    def generate(self, query: str) -> Dict[str, Any]:
        """Generate content synchronously."""
        return self.agent_executor.invoke({"input": query})