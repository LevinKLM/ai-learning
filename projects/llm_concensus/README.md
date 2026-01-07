# Multi-LLM Consensus System (MLCS)

A local-first, multi-agent system designed for deep reasoning and consensus building using local LLMs (Ollama).
A local debate platform where multiple LLMs (`Llama 3.2`, `Mistral NeMo`, `Gemma 2`, `Qwen 2.5`) discuss a topic, critique each other, and reach a consensus, adjudicated by a Judge model.

## Features
-   **Multi-Model Debate**: 4 distinct local models debating in parallel.
-   **Cyclic Consensus**: 3-round debate loop where agents see and critique previous responses.
-   **Judge System**: `Qwen 2.5` acts as the Chief Justice to score responses and pick a winner.
-   **Interactive UI**: Streamlit interface with real-time streaming, expandable debate history, and final verdict.
-   **100% Local**: Runs entirely on your machine via Ollama.

## Setup

### 1. Prerequisites
-   **Ollama**: Install from [ollama.com](https://ollama.com).
-   **Python 3.10+**

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/WSL
pip install -r requirements.txt
```

### 3. Pull Models
Connect to your Ollama instance (ensure `OLLAMA_HOST` is set if needed) and pull the required models:
```bash
ollama pull llama3.2
ollama pull mistral-nemo
ollama pull gemma2
ollama pull qwen2.5:14b
```

### Run CLI
```bash
python main.py
```

### Web UI
Run the Streamlit interface:
```bash
streamlit run src/ui.py
```

## Architecture

-   **`src/state.py`**: Defines the shared state of the debate graph.
-   **`src/models/`**: handlers for LLM inference (Ollama).
-   **`src/graph.py`**: The LangGraph workflow definition (Reflection & Debate loop).
-   **`src/ui.py`**: Minimal frontend for interaction.
