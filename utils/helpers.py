import os
from typing import List, Dict, Any

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "data",
        "vectorstore",
        "config"
    ]
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)


def format_agent_response(response: Dict[str, Any]) -> str:
    """Format the agent response for display."""
    if not response or "output" not in response:
        return "No response or invalid response format."
    
    return response["output"]


def create_env_template():
    """Create a template .env file if it doesn't exist."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    env_path = os.path.join(base_dir, ".env")
    
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("""# Azure OpenAI Configuration
                AZURE_OPENAI_API_KEY=your_api_key_here
                AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
                AZURE_OPENAI_API_VERSION=2023-07-01-preview

                # Azure OpenAI Model Deployments
                AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_model_deployment_name
                AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_model_deployment_name

                # Retrieval Settings
                MAX_RELEVANT_CHUNKS=5
                CHUNK_SIZE=1000
                CHUNK_OVERLAP=200
                """)
        print(f"Created template .env file at {env_path}")
        print("Please update it with your Azure OpenAI credentials.")
    else:
        print(f".env file already exists at {env_path}")