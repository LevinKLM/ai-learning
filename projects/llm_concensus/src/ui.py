import streamlit as st
import asyncio
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import create_graph
from src.models.ollama import MODEL_NAMES

st.set_page_config(page_title="Multi-LLM Consensus System", layout="wide")

st.title("🤖 Multi-LLM Consensus Debate")
st.markdown("A local Council of Experts debating your queries.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    model_list = "\n".join([f"- {name}" for name in MODEL_NAMES.values()])
    st.info(f"Using local Ollama models:\n{model_list}")

user_input = st.text_area("Enter your query:", height=100, placeholder="e.g., Should AI have rights?")

if st.button("Start Debate"):
    if not user_input:
        st.warning("Please enter a query.")
    else:
        status_box = st.status("Thinking...", expanded=True)
        
        # Async runner for Streamlit
        async def run_debate():
            status_box.update(label="Initializing agents...")
            graph = create_graph()
            
            initial_state = {
                "user_input": user_input,
                "context_data": "",
                "responses": {},
                "response_history": [],
                "critiques": [],
                "iteration_count": 0,
                "consensus_reached": False,
                "final_output": ""
            }
            
            status_box.update(label="Running debate rounds...")
            
            # Container to hold the streaming output
            history_container = st.container()
            processed_rounds = 0
            final_state = {}

            # Stream the graph updates
            async for output in graph.astream(initial_state):
                for node_name, node_output in output.items():
                    # capture final state roughly
                    final_state.update(node_output)
                    
                    # If this node produced history (draft_responses or debate_round)
                    if "response_history" in node_output:
                        history = node_output["response_history"]
                        for round_data in history:
                            processed_rounds += 1
                            with history_container:
                                with st.expander(f"Round {processed_rounds}", expanded=True):
                                    cols = st.columns(len(round_data))
                                    for idx, (agent_name, response) in enumerate(round_data.items()):
                                        with cols[idx]:
                                            st.markdown(f"**{agent_name}**")
                                            st.info(response)

                    # If this is the Judge's output
                    if "judge_verdict" in node_output:
                        st.balloons()
                        st.subheader("👨‍⚖️ Final Verdict")
                        st.success(node_output["judge_verdict"])
                        
                        # Extract winner to show Final Report
                        verdict_text = node_output["judge_verdict"]
                        import re
                        match = re.search(r"\*\*Winner\*\*: (.+)", verdict_text)
                        if match:
                            winner_name = match.group(1).strip()
                            # find key for this name
                            winner_key = None
                            for k, v in MODEL_NAMES.items():
                                if v in winner_name or winner_name in v: # loose match
                                    winner_key = k
                                    break
                            
                            if winner_key and winner_key in final_state.get("responses", {}):
                                st.markdown("### 🏆 Final Report (Winner's Answer)")
                                st.markdown(final_state["responses"][winner_key])
                        
            return final_state

        try:
            result = asyncio.run(run_debate())
            status_box.update(label="Debate Complete!", state="complete", expanded=False)
            
            
            # Show raw state details in expander
            with st.expander("View Debug State"):
                st.json(result)
                
        except Exception as e:
            status_box.update(label="Error occurred", state="error")
            st.error(f"An error occurred: {str(e)}")
