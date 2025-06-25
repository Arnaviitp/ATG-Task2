import os
from langchain_core.messages import HumanMessage

from train import fine_tune_model
from nodes import InferenceNode, ConfidenceCheckNode, FallbackNode
from state import State
from utils import log_event

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main_cli(model, tokenizer):
    inference_node = InferenceNode(model, tokenizer)
    confidence_node = ConfidenceCheckNode(threshold=0.75)
    fallback_node = FallbackNode()

    while True:
        user_input = input("\nEnter text for classification (or type 'exit' to quit):\n> ").strip()
        if user_input.lower() == "exit":
            print("Exiting.")
            break

        state: State = {
            "messages": [HumanMessage(content=user_input)],
            "classification": "",
            "confidence": 0.0,
            "input_text": user_input,
            "fallback_triggered": False,
            "clarification": None,
        }

        state = inference_node(state)
        result = confidence_node(state)
        if result is None:  # END
            log_event(f"Final Label: {state['classification']} (Accepted, Confidence: {state['confidence']*100:.1f}%)")
            continue
        state = fallback_node(state)
        log_event(f"Final Label: {state['classification']} (Corrected via user clarification)")

if __name__ == "__main__":
    if not os.path.exists("./finetuned_model"):
        print("Fine-tuned model not found. Training now...")
        model, tokenizer = fine_tune_model()
    else:
        model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model")
        tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")

    main_cli(model, tokenizer)
