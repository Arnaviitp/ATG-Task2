import torch
from langchain_core.messages import HumanMessage
from typing import Union
from langgraph.graph import END
from typing import Union

from utils import log_event
from state import State

class InferenceNode:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, state: State) -> State:
        text = state["input_text"]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_label = int(probs.argmax())
            confidence = float(probs[pred_label])
            label_map = {0: "Negative", 1: "Positive"}
            classification = label_map[pred_label]
        log_event(f"[InferenceNode] Predicted label: {classification} | Confidence: {confidence*100:.1f}%")
        state["classification"] = classification
        state["confidence"] = confidence
        state["fallback_triggered"] = False
        return state

class ConfidenceCheckNode:
    def __init__(self, threshold=0.75):
        self.threshold = threshold
    from typing import Optional

    def __call__(self, state: State) -> Optional[State]:

        if state["confidence"] < self.threshold:
            log_event("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
            state["fallback_triggered"] = True
            return state
        else:
            log_event(f"[ConfidenceCheckNode] Confidence sufficient. Accepting prediction.")
            return END

class FallbackNode:
    def __call__(self, state: State) -> State:
        clarification_prompt = f"Could you clarify your intent? Was this a {state['classification'].lower()} review?"
        log_event(f"[FallbackNode] {clarification_prompt}")
        clarification = input(f"{clarification_prompt}\nUser: ").strip()
        if clarification.lower() in ["yes", "y"]:
            log_event("[FallbackNode] User confirmed original classification.")
        else:
            new_label = "Negative" if state["classification"] == "Positive" else "Positive"
            log_event(f"[FallbackNode] User correction: Changing label to {new_label}.")
            state["classification"] = new_label
        state["clarification"] = clarification
        return state
