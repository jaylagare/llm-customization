from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# === Load base model and tokenizer ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Required for training

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# === Load and format Alpaca dataset ===
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

def format_alpaca(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

dataset = dataset.map(format_alpaca)

# === Tokenize dataset ===
def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# === Set up Trainer ===
training_args = TrainingArguments(
    output_dir="./tinyllama-full",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    fp16=True,  # Keep True for GPU
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Train the model ===
trainer.train()

# === Save fine-tuned model and tokenizer ===
model.save_pretrained("tinyllama-alpaca-full")
tokenizer.save_pretrained("tinyllama-alpaca-full")