import asyncio
import sys
import logging
from src.graph import create_graph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

async def main():
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = "What is the meaning of justice?"

    print(f"Running debate on: '{user_input}'")
    print("Initializing agents... (this may take a moment)")
    
    graph = create_graph()
    
    initial_state = {
        "user_input": user_input,
        "context_data": "",
        "responses": {},
        "critiques": [],
        "iteration_count": 0,
        "consensus_reached": False,
        "final_output": ""
    }

    print("Thinking... (Logs should appear below)")
    # Invoke the graph
    result = await graph.ainvoke(initial_state)
    
    print("\n" + "="*50 + "\n")
    print(result.get("final_output", "No output generated."))
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
