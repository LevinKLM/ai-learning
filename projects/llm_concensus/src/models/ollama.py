from langchain_ollama import ChatOllama

# Define the models we want to use
# We can make this configurable later, but hardcoding for Phase 1 as per plan
MODEL_NAMES = {
    "model_a": "llama3.2",
    "model_b": "mistral-nemo",
    "model_c": "gemma2",
    "model_d": "qwen2.5:14b",
}

import os

def get_model(model_name: str, temperature: float = 0.7):
    """
    Returns a configured ChatOllama instance.
    """
    # Default to localhost if not set, but allow override
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url
    )

def init_models():
    """
    Initializes and returns a dictionary of the 3 agents.
    """
    agents = {}
    for agent_id, model_name in MODEL_NAMES.items():
        agents[agent_id] = get_model(model_name)
    return agents
