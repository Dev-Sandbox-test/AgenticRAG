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


def main():
    parser = argparse.ArgumentParser(description="RAG System with LangChain Agents")
    parser.add_argument("query", nargs="?", default=None, help="The query to run against the RAG system")
    parser.add_argument("--data-dir", "-d", help="Directory containing documents to index")
    parser.add_argument("--refresh", "-r", action="store_true", help="Force refresh the vector store")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("=== RAG System Interactive Mode ===")
        print("Type 'exit' to quit.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == "exit":
                break
            asyncio.run(setup_and_run_query(query, args.data_dir, args.refresh))
            # Only refresh on first query if specified
            args.refresh = False
    elif args.query:
        asyncio.run(setup_and_run_query(args.query, args.data_dir, args.refresh))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()