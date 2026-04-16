"""
test_graph.py - Test the LangGraph pipeline end to end
Run from backend/: python test_graph.py
"""
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from memory.long_term import initialize_memory_db
from memory.short_term import create_initial_state, format_state_for_display
from graph.graph import get_graph

def test_graph():
    print("\n" + "="*60)
    print("Testing LangGraph Pipeline")
    print("="*60)

    # Initialize memory database
    initialize_memory_db()
    print("Memory DB initialized")

    # Create the graph
    graph = get_graph()
    print("Graph compiled")

    # Create initial state
    query = "What are recent AI approaches for detecting Alzheimer's disease?"
    state = create_initial_state(query)
    session_id = state["session_id"]

    print(f"\nQuery: {query}")
    print(f"Session ID: {session_id[:8]}...")
    print("\nRunning graph (will pause at human checkpoint)...\n")

    # Configuration for LangGraph
    # thread_id is how LangGraph identifies this run
    # It's used to save/load the checkpoint
    config = {"configurable": {"thread_id": session_id}}

    # Run the graph - it will pause before human_checkpoint_node
    events = graph.stream(state, config)

    for event in events:
        # Each event is {node_name: state_update}
        node_name = list(event.keys())[0]
        print(f"Node executed: {node_name}")

        # Print trace entries as they're added
        if "trace" in event.get(node_name, {}):
            for entry in event[node_name]["trace"]:
                print(f"  TRACE: {entry}")

    # After streaming, get the current state
    current_state = graph.get_state(config)

    print("\n" + "="*60)
    print("Graph paused at human checkpoint")
    print("="*60)

    display = format_state_for_display(dict(current_state.values))
    print(f"Status:     {display['status']}")
    print(f"Tools used: {display['tools_used']}")
    print(f"\nDRAFT ANSWER (first 500 chars):")
    print("-"*40)
    draft = display.get('draft_answer', 'No draft')
    print(draft[:500] if draft else "No draft generated")

    print("\n" + "="*60)
    print("Simulating human approval...")
    print("="*60)

    # Simulate human approval by updating state and resuming
    graph.update_state(
        config,
        {"human_approved": True},
    )

    # Resume the graph after approval
    final_events = graph.stream(None, config)
    for event in final_events:
        node_name = list(event.keys())[0]
        print(f"Node executed: {node_name}")

    # Get final state
    final_state = graph.get_state(config)
    final_display = format_state_for_display(dict(final_state.values))

    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    final = final_display.get('final_answer', 'No final answer')
    print(final[:800] if final else "No final answer generated")
    print("\nTest complete!")

if __name__ == "__main__":
    test_graph()