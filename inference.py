from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the upgraded model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_response(ai_name: str, ai_task: str, user_prompt: str) -> str:
    full_prompt = (
        f"You are {ai_name}, a helpful AI assistant. "
        f"Your job is to {ai_task}. "
        f"Answer clearly and completely.\n\n"
        f"Question: {user_prompt}\n"
        f"Answer:"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        return decoded.split("Answer:")[-1].strip()
    else:
        return decoded.strip()
