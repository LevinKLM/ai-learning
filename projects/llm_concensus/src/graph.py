from typing import Dict, List, Annotated
import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

import logging

# Configure logging
logger = logging.getLogger(__name__)

from src.state import ConsensusState
from src.models.ollama import init_models, MODEL_NAMES

# Initialize models once
AGENTS = init_models()

SYSTEM_PROMPT = """You are a helpful assistant participating in a debate. 
Your goal is to provide a comprehensive and accurate answer to the user's query.
"""

DEBATE_PROMPT_TEMPLATE = """
User Query: {user_input}

Current Context / Previous Round Responses:
{previous_responses}

Instruction:
Reflect on the responses above. Identify any discrepancies, valid points, or errors in reasoning.
Then, provide an updated and improved answer to the original query, incorporating valid insights from others while maintaining your own correct perspective.
"""

JUDGE_PROMPT_TEMPLATE = """
User Query: {user_input}

Final Responses from Experts:
{final_responses}

Instruction:
You are the Chief Justice. Your task is to evaluate the responses above.
1. Assign a numeric score (0-10) to EACH expert based on accuracy, depth, and clarity.
2. Select the single BEST response as the Winner.
3. Provide a justification for your choices.

Format your output EXACTLY as follows:

## Verdict
**Winner**: [Winning Model Name]
**Reason**: [Brief explanation]

### Scores
- [Model Name]: [Score]/10
- [Model Name]: [Score]/10
...
"""

async def draft_responses(state: ConsensusState):
    """
    Node: Generates initial responses from all agents in parallel.
    """
    user_input = state["user_input"]
    context = state.get("context_data", "")
    
    prompt_content = f"User Query: {user_input}\nContext: {context}"
    
    logger.info(f"Drafting responses for query: {user_input}")

    # Define an async task for each agent
    async def call_model(name: str, model):
        logger.info(f"[{name}] Starting generation...")
        # Using invoke for simplicity in this phase, 
        # but in a real async loop we might want astream or batch
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        try:
            response = await model.ainvoke(messages)
            logger.info(f"[{name}] Generation complete.")
            return name, response.content
        except Exception as e:
            logger.error(f"[{name}] Generation failed: {e}")
            return name, f"Error: {e}"

    # Run all models concurrently
    tasks = [call_model(name, agent) for name, agent in AGENTS.items()]
    results = await asyncio.gather(*tasks)
    
    # Update state
    # Since we are using a reducer for iteration_count, we just emit 1 to add to the total.
    # We also pass the new responses.
    
    new_responses = {name: content for name, content in results}

    return {
        "responses": new_responses,
        "response_history": [new_responses], # Append as a new item in the history list
        "iteration_count": 1 
    }

async def debate_round(state: ConsensusState):
    """
    Node: Review phase. Models see each other's answers and update their own.
    """
    user_input = state["user_input"]
    current_responses = state["responses"]
    
    # Format previous responses for context
    formatted_responses = "\n".join([f"[{name}]: {resp}" for name, resp in current_responses.items()])
    
    prompt_content = DEBATE_PROMPT_TEMPLATE.format(
        user_input=user_input,
        previous_responses=formatted_responses
    )
    
    logger.info(f"Starting Debate Round (Current Iteration: {state['iteration_count']})")

    # Define an async task for each agent
    async def call_model(name: str, model):
        logger.info(f"[{name}] Refuting/Updating...")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        try:
            response = await model.ainvoke(messages)
            logger.info(f"[{name}] Update complete.")
            return name, response.content
        except Exception as e:
            logger.error(f"[{name}] Update failed: {e}")
            return name, f"Error: {e}"

    # Run all models concurrently
    tasks = [call_model(name, agent) for name, agent in AGENTS.items()]
    results = await asyncio.gather(*tasks)
    
    new_responses = {name: content for name, content in results}
    
    return {
        "responses": new_responses,
        "response_history": [new_responses],
        "iteration_count": 1
    }

async def judge_round(state: ConsensusState):
    """
    Node: The Judge evaluates the final responses.
    We use the strongest model (Model D - Qwen 2.5 14B) as the Judge.
    """
    user_input = state["user_input"]
    current_responses = state["responses"]
    
    # Use real model names for the Judge context
    from src.models.ollama import MODEL_NAMES
    formatted_responses = "\n".join([f"[{MODEL_NAMES[name]}]: {resp}" for name, resp in current_responses.items()])
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        user_input=user_input,
        final_responses=formatted_responses
    )
    
    logger.info("👨‍⚖️ Judge is deliberating...")
    
    # Use Model D (Qwen) as Judge
    judge_model = AGENTS["model_d"]
    
    messages = [
        SystemMessage(content="You are an impartial and wise judge."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await judge_model.ainvoke(messages)
        verdict = response.content
        logger.info("Verdict reached.")
    except Exception as e:
        logger.error(f"Judge failed: {e}")
        verdict = f"Error in judgement: {e}"
        
    return {"judge_verdict": verdict}

def should_continue(state: ConsensusState):
    """
    Conditional Edge: Checks if we reached max iterations.
    """
    # Run for 3 rounds total (Initial + 2 revisions)
    if state["iteration_count"] >= 3: 
        return "finalize"
    return "debate"

async def finalize_response(state: ConsensusState):
    """
    Node: Synthesize or just return the responses.
    """
    # Just pass through, UI handles display. 
    # Can also append verdict to final output if CLI usage.
    verdict = state.get("judge_verdict", "")
    return {"final_output": verdict}

def create_graph():
    workflow = StateGraph(ConsensusState)

    # Add Nodes
    workflow.add_node("draft_responses", draft_responses)
    workflow.add_node("debate_round", debate_round)
    workflow.add_node("finalize_response", finalize_response)

    # Set Entry Point
    workflow.set_entry_point("draft_responses")

    # Add Edges
    workflow.add_edge("draft_responses", "debate_round") # Start the loop
    
    # Conditional logic after debate_round
    workflow.add_conditional_edges(
        "debate_round",
        should_continue,
        {
            "debate": "debate_round",
            "finalize": "judge_round"
        }
    )
    
    workflow.add_node("judge_round", judge_round)
    workflow.add_edge("judge_round", "finalize_response")
    workflow.add_edge("finalize_response", END)

    return workflow.compile()
