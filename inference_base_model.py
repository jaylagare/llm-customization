from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for training

# Choose device manually
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model on the selected device
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
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

print("=== BEFORE Fine-Tuning ===")
print(generate(model, prompt))