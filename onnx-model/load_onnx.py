import onnxruntime as ort
import numpy as np
import os
from config import Config

def load_and_run_onnx_model(onnx_model_path, input_data):
    sess = ort.InferenceSession(onnx_model_path)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    outputs = sess.run([output_name], {input_name: input_data})

    return outputs[0]

if __name__ == '__main__':
    model_path = os.path.join(Config["model_path"],"model_%s.onnx"%(Config["model_name"]))
    batch_size = 1
    seq_len = Config["max_length"]
    dummy_input = np.zeros((batch_size, seq_len), dtype=np.int64)
    output = load_and_run_onnx_model(model_path, dummy_input)
    print("Model output:", output)