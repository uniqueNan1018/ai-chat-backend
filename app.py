from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)  # ここでCORS対応

model_name = "gpt2"  # GPT-2モデルを指定

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("cpu")  # CPUで動かす場合

@app.route("/chat", methods=["POST", "OPTION"])
def chat():
    user_input = request.json.get("message", "")
    prompt = f"User: {user_input}\nAI:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=40,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.7,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            early_stopping=True
        )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply.replace(prompt, "").strip()

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)