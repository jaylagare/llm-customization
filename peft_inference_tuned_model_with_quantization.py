from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load base model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
peft_model_path = "tinyllama-alpaca-lora"  # Path to your LoRA fine-tuned adapter

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# BitsAndBytes quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None
)

# Load base model with 8-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)
base_model.eval()

# Load PEFT model
model = PeftModel.from_pretrained(base_model, peft_model_path)
model.eval()

# Generate function
def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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