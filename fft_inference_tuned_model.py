from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model name — make sure this matches what you used earlier
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_path = "tinyllama-alpaca-full"  # Path to your full fine-tuned model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Choose device manually
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

# Load fully fine-tuned model
model = AutoModelForCausalLM.from_pretrained(model_path)
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

print("=== AFTER Full Fine-Tuning ===")
print(generate(model, prompt))