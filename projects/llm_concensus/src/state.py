from typing import TypedDict, Dict, List, Optional, Annotated
import operator

class ConsensusState(TypedDict):
    """
    State dictionary to track the consensus process.
    """
    user_input: str
    # The original query or topic provided by the user.

    context_data: Optional[str]
    # Raw text extracted from any provided documents (PDFs) or image descriptions.

    responses: Dict[str, str]
    # A dictionary mapping model names to their current argument/response.
    # We use default replacement for dicts in TypedDict, but explicit Annotated is allowed.
    # For now, standard behavior (overwrite key usage) is fine, but let's stick to simple TypedDict
    # for responses if we just overwrite. 
    # USER ASKED FOR ANNOTATED:
    # Let's make responses accumulate/merge updates.
    
    critiques: Annotated[List[str], operator.add]
    # A list of criticisms. Use operator.add to append new items.

    response_history: Annotated[List[Dict[str, str]], operator.add]
    # A list of dictionaries, where each dict contains the responses for a single round.
    # index 0 = Round 1 (Draft), index 1 = Round 2 (Debate 1), etc.

    iteration_count: Annotated[int, operator.add]
    # Tracks the current round. We will emit '1' to increment.

    judge_verdict: Optional[str]
    # The final evaluation from the Judge model.
    
    consensus_reached: bool
    final_output: Optional[str]
