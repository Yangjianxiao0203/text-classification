from flask import Flask, render_template, request, jsonify
import logging
import os
from onnx_model.load_onnx import classify_text
import json
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = os.path.join("./model","model_bert_cnn_heavy.onnx")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    text = request.get_json()["text"]
    probs = classify_text(text,model_path)
    logger.info(f"Text: {text}, Probabilities: {probs}")
    display_json = jsonify(probs)
    return display_json

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)