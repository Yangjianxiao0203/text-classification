import onnxruntime as ort
import numpy as np
import os
from config import Config
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
labels = {
    0:"sadness",
    1:"joy",
    2:"love",
    3:"anger",
    4:"fear",
    5:"surprise"
}

def load_and_run_onnx_model(onnx_model_path, input_data):
    sess = ort.InferenceSession(onnx_model_path)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    input_data = input_data.reshape(1,-1).numpy()

    input_feed = {input_name: input_data}

    outputs = sess.run([output_name], input_feed)

    return outputs[0].squeeze()

def classify_text(text,model_path):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        truncation=True,
        padding="max_length",
        return_attention_mask=False,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].flatten().clone().detach()
    output = load_and_run_onnx_model(model_path, input_ids)
    # softmax
    output = np.exp(output) / np.sum(np.exp(output))
    probs = {}
    for i in range(len(output)):
        probs[labels[i]] = float(output[i])
    return probs


if __name__ == '__main__':
    model_path = os.path.join("../model","model_%s.onnx"%(Config["model_name"]))
    text = "I am very sad but wonder if we could go to Beijing."
    emotion = classify_text(text,model_path)
    print(emotion)