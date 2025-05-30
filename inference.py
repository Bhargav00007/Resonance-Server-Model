from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ✅ Use Hugging Face cache location supported on Render
cache_dir = "/opt/render/project/.cache"

# ✅ Load lightweight model for Render Free Tier (~400MB RAM)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir=cache_dir)

def generate_response(ai_name: str, ai_task: str, user_prompt: str) -> str:
    """
    Generates a response based on AI assistant name, task, and user input.
    """
    full_prompt = (
        f"You are {ai_name}, a helpful AI assistant. "
        f"Your job is to {ai_task}. "
        f"Answer clearly and completely.\n\n"
        f"Question: {user_prompt}\n"
        f"Answer:"
    )

    # Tokenize user input
    inputs = tokenizer(full_prompt, return_tensors="pt")

    # Generate model output
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # ⬅️ Keep this small for memory safety
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    # Decode and clean response
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the AI's answer
    if "Answer:" in decoded:
        return decoded.split("Answer:")[-1].strip()
    else:
        return decoded.strip()
