from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)  # ここでCORS対応

model_name = "gpt2"  # GPT-2モデルを指定

# モデルとトークナイザーの準備
# GPT-2のモデルとトークナイザーをロード。
# to("cpu") により、GPUがない環境でも動くようにCPUで推論。
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("cpu")  # CPUで動かす場合

@app.route("/chat", methods=["POST", "OPTION"])
def chat():
    user_input = request.json.get("message", "")
    prompt = f"User: {user_input}\nAI:"
    # トークナイズしてモデルに渡す
    inputs = tokenizer(prompt, return_tensors="pt") # 入力をトークンに変換。
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # モデルのデバイス（この場合はCPU）に移動。
    # テキスト生成
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
    # 出力トークンをテキストに変換。
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # プロンプト部分を削除し、AIの返答だけを取り出す。
    reply = reply.replace(prompt, "").strip()

    return jsonify({"reply": reply})

# このPythonファイルを直接実行したときに、Flaskサーバーを起動。
if __name__ == "__main__":
    app.run(debug=True)