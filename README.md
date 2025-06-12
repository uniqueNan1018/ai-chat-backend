# Flask チャットAPI（GPT-2 使用）

このプロジェクトは、Python・Flask・Hugging Face Transformersライブラリを使って構築されたシンプルなチャットボットAPIです。  
GPT-2モデルを使用して、ユーザーのメッセージに対して自動で返信を生成します。

---

## 主な機能

- Flaskで構築されたREST API
- GPT-2を使った自然言語生成
- CORS対応（他のドメインからもアクセス可能）
- CPU環境で動作（GPU不要）
- 応答生成のパラメータ調整可能（top_p、top_k、temperatureなど）

---

## 動作環境

- Python 3.8以上
- pip パッケージマネージャー

### 必要なPythonライブラリ

以下のコマンドで依存パッケージをインストールします：

```bash
pip install flask flask-cors transformers torch
```

## ファイル構成
```bash
chatbot/
├── app.py         # Flask アプリ本体
└── README.md      # この説明ファイル
```

## 実行方法
- 1. ライブラリのインストール
```bash
pip install flask flask-cors transformers torch
```
- 2. Flaskサーバーの起動
```bash
python app.py
```
起動後、http://localhost:5000 でAPIが使用可能になります。

## APIの使い方
- 1. エンドポイント
```bash
POST /chat
```
- 2. リクエストヘッダー
```bash
Content-Type: application/json
```
- 3. リクエストボディの例
```json
{
  "message": "こんにちは、調子はどう？"
}
```
- 4. レスポンスボディの例
```json
{
  "reply": "こんにちは！元気ですよ。あなたはどうですか？"
}
```

## APIの使い方
```python
outputs = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.8,
    top_k=30,
    temperature=0.7,
    no_repeat_ngram_size=2,
    early_stopping=True
)
```
### top_p, top_k, temperature などを変更することで、出力の自然さや多様性を調整できます。

## 注意事項
- gpt2 は英語モデルなので、日本語には適していません。
日本語対応モデル（例：rinna/japanese-gpt-1b や ELYZA-japanese-Llama-2-7b-instruct）を使用することで、より自然な応答になります。

- CPU環境では処理が重くなる場合があります。可能であればGPU環境を推奨します。



