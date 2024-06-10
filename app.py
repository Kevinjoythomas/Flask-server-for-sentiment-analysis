from flask import Flask,request,jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__)
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
def to_numpy(tensor):
    return{
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    }

@app.route("/")
def home():
    return "<h2>Robertas sentiment analysis</h2>"

@app.route("/predict",methods = ["POST"])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error':'No input text'}),400
    input_text = data['text']
    input = tokenizer(input_text,return_tensors = "pt",padding=True,truncation =True)
    with torch.no_grad():
        outputs = model(**input)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    sentiment = "positive" if prediction == 2 else "neutral" if prediction == 1 else "negative"
    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)