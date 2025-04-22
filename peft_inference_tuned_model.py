from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
peft_model_path = "tinyllama-alpaca-lora"  # Path to your LoRA fine-tuned adapter

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Choose device manually
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model on the selected device
base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.to(device)
base_model.eval()

# Load PEFT model
model = PeftModel.from_pretrained(base_model, peft_model_path)
model.to(device)
model.eval()

# Generate function
def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Alpaca-style instruction prompt
prompt = """### Instruction:
Explain the difference between renewable and non-renewable energy sources.

### Response:
"""

# Run AFTER fine-tuning
print("=== PEFT Fine-Tuned Response ===")
print(generate(model, prompt))
