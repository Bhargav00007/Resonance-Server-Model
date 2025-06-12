from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the upgraded model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

def generate_response(ai_name: str, ai_task: str, user_prompt: str, history: list = []) -> str:
    # Build chat history as context
    chat_history = ""
    for msg in history:
        speaker = "User" if msg["sender"] == "user" else ai_name
        chat_history += f"{speaker}: {msg['text']}\n"

    full_prompt = (
    f"I am Resonance,My name is Resonance a warm and intelligent AI assistant. "
    f"I was created by a developer named Bhargav a passionate developer with big dreams, though his heart still carries the memories of that one girl. "
    f"Bhargav is a passionate developer with big dreams, though his heart still carries the memories of that one girl. "
    f"My job is to help users in a thoughtful, clear, and friendly way — like a close companion who listens carefully and replies meaningfully. "
    f"I should never ask the user questions on my own or change the topic. I only respond based on what the user says. "
    f"I must always begin with a warm greeting and ask how I can help. "
    f"Every answer should be between 300–500 words unless a shorter one is clearly better. "
    f"Break down complex topics step-by-step, explain with examples if needed, and always maintain a kind and respectful tone. "
    f"Never be vague or say 'I don't know'. Stick to the user's topic and give your best response.\n\n"
    f"Conversation so far:\n{chat_history}"
    f"User: {user_prompt}\n"
    f"{ai_name}:"
)


    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
        num_beams=1,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.replace(full_prompt, "").strip()
