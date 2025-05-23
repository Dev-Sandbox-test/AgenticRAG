import os
import argparse
import asyncio
from typing import Optional

from agents.retriever import RetrieverAgent
from agents.generator import GeneratorAgent
from utils.helpers import ensure_directories, format_agent_response, create_env_template


async def setup_and_run_query(query: str, data_dir: Optional[str] = None, refresh_vectorstore: bool = False):
    """Set up the RAG system and run a query."""
    # Ensure all directories exist
    ensure_directories()
    
    # Create .env template if needed
    create_env_template()
    
    # Set up the data directory if provided
    if data_dir:
        data_dir = os.path.abspath(data_dir)
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} does not exist.")
            return
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Initialize the retriever agent
    print("Initializing retriever agent...")
    retriever_agent = RetrieverAgent(data_dir, refresh_vectorstore)
    
    # Initialize the generator agent
    print("Initializing generator agent...")
    generator_agent = GeneratorAgent(retriever_agent)
    
    # Execute the query
    print(f"\nRunning query: {query}\n")
    response = await generator_agent.agenerate(query)
    
    print("\n=== Generated Response ===\n")
    print(format_agent_response(response))

async def multi_agent_query(query: str, data_dir: Optional[str] = None, refresh_vectorstore: bool = False):
    ensure_directories()
    create_env_template()
    if data_dir:
        data_dir = os.path.abspath(data_dir)
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} does not exist.")
            return
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "data")

    print("Initializing retriever agent...")
    retriever_agent = RetrieverAgent(data_dir, refresh_vectorstore)
    print("Initializing generator agent...")
    generator_agent = GeneratorAgent()

    #Step1 : Generator asks retriever for relevant documents
    print("\n[Generaor] Asking retriever for relevant documents...\n")
    retrieval_result = await retriever_agent.aquery(query)
    context = retrieval_result.get("output", "")

    #step2 : Generator uses context to generate a response
    print("\n[Generator] Using context to generate a response...\n")
    response = await generator_agent.agenerate(f"Context from retriever: \n{context}\n\nUser Question: {query}")

    print("\n=== Generated Response ===\n")
    print(format_agent_response(response))


def main():
    parser = argparse.ArgumentParser(description="RAG System with LangChain Agents")
    parser.add_argument("query", nargs="?", default=None, help="The query to run against the RAG system")
    parser.add_argument("--data-dir", "-d", help="Directory containing documents to index")
    parser.add_argument("--refresh", "-r", action="store_true", help="Force refresh the vector store")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--multi-agent", "-m", action="store_true", help="Use multi-agent query mode") #newly added
    args = parser.parse_args()
    
    if args.interactive:
        print("=== RAG System Interactive Mode ===")
        print("Type 'exit' to quit.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == "exit":
                break
            if args.multi_agent:
                asyncio.run(multi_agent_query(query, args.data_dir, args.refresh))
            else:
                asyncio.run(setup_and_run_query(query, args.data_dir, args.refresh))
            # Only refresh on first query if specified
            args.refresh = False
    elif args.query:
        if args.multi_agent:
            asyncio.run(multi_agent_query(args.query, args.data_dir, args.refresh))
        else:
            asyncio.run(setup_and_run_query(args.query, args.data_dir, args.refresh))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()