import json
import os
import re
from collections import Counter

from dotenv import load_dotenv
import google.generativeai as genai

# Gemini API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing API key! Set GOOGLE_API_KEY in your .env file.")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def clean_json_response(text):
    # Try to extract JSON content between backticks if present
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    
    # If no backticks found, try to find JSON object between { and }
    json_pattern = r'(\{[\s\S]*\})'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    
    # Otherwise return the original text
    return text.strip()

def generate_responses(user_prompt, model="gemini-2.0-flash-001", num_samples=3, temperature=0.7):
    model = genai.GenerativeModel(model)
    responses = []
    system_prompt = """
    You are an expert math tutor. Always explain step-by-step solutions in a clear and structured manner.
    Your response MUST be valid JSON without any text before or after the JSON object.
    Do not include markdown formatting, code blocks, or any other text.
    Format your response exactly like this:

    {
        "solution": [
            "Step 1: [explanation]",
            "Step 2: [explanation]",
            ...
        ],
        "answer": "Final numerical result"
    }
    """
    prompt = system_prompt + "\n\n" + user_prompt
    for _ in range(num_samples):
        try:
            response = model.generate_content(
                [{"role": "user", "parts": [prompt]}],
                generation_config=genai.types.GenerationConfig(temperature=temperature, max_output_tokens=1000)
            )
            response_text = clean_json_response(response.text)
            json_data = json.loads(response_text)  # Check if the response is valid JSON
            responses.append(json_data)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return responses

def self_consistency(prompt, num_samples=5):
    responses = generate_responses(prompt, num_samples=num_samples)
    final_answers = []
    for response in responses:
        if response is not None:
            final_answers.append(response["answer"])

    if final_answers:
        most_common = Counter(final_answers).most_common(1)[0][0]  # Get most frequent answer
    else:
        most_common = "No valid answers found"
    return most_common, responses

# Example Prompt
prompt = "A train leaves City A at 10:00 AM, traveling at 60 km/h toward City B, which is 180 km away. Another train leaves City B at 11:00 AM, traveling at 90 km/h toward City A. At what time will they meet?"

final_answer, all_responses = self_consistency(prompt)

print("All Responses:")

# Display each response
for index, response in enumerate(all_responses, start=1):
    print(f"\nResponse {index}:")
    for line in response.get("solution", []):  # Iterate through solution steps
        print(line)
    print(f"Final Answer: {response.get('answer', 'No answer found')}")

# Display final answer
print("\nFinal Answer (Majority Vote):", final_answer)