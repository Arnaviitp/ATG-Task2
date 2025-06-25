# ATG Task 2

```markdown
# Self-Healing Classification DAG with Fine-Tuned Transformer

This project implements a robust, self-healing text classification pipeline using a fine-tuned DistilBERT model (with LoRA), and a modular Directed Acyclic Graph (DAG) workflow using LangGraph. The system automatically handles low-confidence predictions by asking users for clarification, ensuring trustworthy outputs.

---

## Features

- **Fine-tuned Transformer**: DistilBERT fine-tuned on SST-2 sentiment dataset using LoRA.
- **DAG Workflow**: Modular nodes for inference, confidence checking, and user fallback, implemented with LangGraph.
- **User-in-the-Loop**: Low-confidence predictions trigger a user clarification prompt.
- **Structured Logging**: All predictions, confidence scores, and user interactions are logged.
- **CLI Interface**: Simple command-line interface for demonstration and testing.

---

## Directory Structure

```
classification_dag/
├── main.py              # Entry point: CLI and workflow logic
├── train.py             # Model fine-tuning logic
├── nodes.py             # All DAG node classes
├── state.py             # State definition for the workflow
├── utils.py             # Logging and utilities
├── requirements.txt     # All dependencies
├── README.md            # This file
└── finetuned_model/     # Saved model after training (created after first run)
```

---

## Installation

1. **Clone the repository:**
   ```
   git clone 
   cd classification_dag
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

---

## Fine-Tuning Instructions

Before using the classifier, you need a fine-tuned model. This project will automatically fine-tune DistilBERT with LoRA on the SST-2 dataset if no model is found in `finetuned_model/`.

**To manually run fine-tuning:**

1. Open a terminal in the project directory.
2. Run:
   ```
   python train.py
   ```
   - This will download the SST-2 dataset, fine-tune DistilBERT using LoRA, and save the model and tokenizer in the `finetuned_model/` directory.
   - The process may take several minutes depending on your hardware.

**Note:**  
If you run `python main.py` and no model is found, fine-tuning will be triggered automatically.

---

## Launching and Interacting with the LangGraph DAG

The workflow is built using LangGraph, which manages the following nodes:

1. **InferenceNode:** Runs the model prediction and computes confidence.
2. **ConfidenceCheckNode:** Checks if confidence is above a set threshold.
3. **FallbackNode:** If confidence is low, asks the user for clarification and updates the label if necessary.

**To launch the CLI and interact with the DAG:**

```
python main.py
```

- If the model is not yet fine-tuned, this will trigger training first.
- Once the model is ready, the CLI will prompt for input text.

---

## CLI Flow Explanation

The CLI guides you through the classification process:

1. **Input Prompt:**  
   - The CLI asks:  
     ```
     Enter text for classification (or type 'exit' to quit):
     > 
     ```
   - Enter any sentence for sentiment classification.

2. **Inference and Confidence Check:**  
   - The system predicts the sentiment (Positive/Negative) and displays the confidence.
   - If the confidence is above the threshold (default 0.75), the result is accepted and logged.
   - If confidence is low, the system asks for clarification.

3. **Fallback/User Clarification:**  
   - Example prompt:  
     ```
     [FallbackNode] Could you clarify your intent? Was this a positive review?
     User: 
     ```
   - Respond with `yes`/`y` to confirm the model’s prediction, or `no`/`n` to flip it.
   - The final label is updated and logged.

4. **Exit:**  
   - Type `exit` at any prompt to quit the CLI.

**Example CLI Session:**

```
Enter text for classification (or type 'exit' to quit):
> The movie was surprisingly good!

[InferenceNode] Predicted label: Positive | Confidence: 88.3%
[ConfidenceCheckNode] Confidence sufficient. Accepting prediction.
Final Label: Positive (Accepted, Confidence: 88.3%)
```

If confidence is low:
```
[InferenceNode] Predicted label: Negative | Confidence: 62.1%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?
User: no
[FallbackNode] User correction: Changing label to Positive.
Final Label: Positive (Corrected via user clarification)
```

---

## Customization

- **Dataset**: Change the dataset in `train.py` to use your own data.
- **Confidence Threshold**: Adjust the threshold in `nodes.py` (`ConfidenceCheckNode`).
- **Model**: Swap DistilBERT for any HuggingFace transformer model.

---

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed and up to date.
- **peft Errors**: Use `prepare_model_for_kbit_training` instead of deprecated functions.
- **transformers Errors**: Use `eval_strategy` (not `evaluation_strategy`) if using transformers < 4.52.

---

## References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [SST-2 Dataset](https://huggingface.co/datasets/sst2)

---

## License

MIT License

**Developed by Arnav Anand, IIT Patna**
