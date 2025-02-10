
from graph import create_graph
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END


class UserInput(TypedDict):
    input: str
    continue_conversation: bool

def get_user_input(state: UserInput) -> UserInput:
    try:
        user_input = input("\nEnter your question (or 'q' to quit) : ")
        return {
            "input": user_input,
            "continue_conversation": user_input.lower() != 'q'
        }
    except Exception as e:
        print("get_user_input error")
        print(e)
        return None

def process_question(state: UserInput):
    try:
        graph = create_graph()
        result = graph.invoke({"input": state["input"]})
        print("\n--- Final answer ---")
        print(result["output"])
        return state
    except Exception as e:
        print("process_question error")
        print(e)
        return None

def create_conversation_graph():
    try:
        workflow = StateGraph(UserInput)

        workflow.add_node("get_input", get_user_input)
        workflow.add_node("process_question", process_question)

        workflow.set_entry_point("get_input")

        workflow.add_conditional_edges(
            "get_input",
            lambda x: "continue" if x["continue_conversation"] else "end",
            {
                "continue": "process_question",
                "end": END
            }
        )

        workflow.add_edge("process_question", "get_input")

        return workflow.compile()
    except Exception as e:
        print("create_conversation_graph error")
        print(e)
        return None

def main():
    try:
        conversation_graph = create_conversation_graph()

        conversation_graph.invoke({"input": "", "continue_conversation": True})
    except Exception as e:
        print("main error")
        print(e)
        return None

if __name__ == "__main__":
    main()