import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def fine_tune_model():
    model_ckpt = "distilbert-base-uncased"
    dataset = load_dataset("sst2")

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    def tokenize_fn(example):
        return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)
    dataset = dataset.map(tokenize_fn, batched=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none"
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].shuffle(seed=42).select(range(1000)),
        eval_dataset=dataset["validation"].select(range(200)),
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    return model, tokenizer
