from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import generate_response

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    name = data.get("name", "Resonance")
    task = data.get("task", "answering all kinds of questions")
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = generate_response(name, task, prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
