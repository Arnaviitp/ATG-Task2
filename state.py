from typing import Annotated, TypedDict, Optional, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    classification: str
    confidence: float
    input_text: str
    fallback_triggered: bool
    clarification: Optional[str]
