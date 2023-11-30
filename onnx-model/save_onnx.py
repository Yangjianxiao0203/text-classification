from model import BertCNNHeavy
import torch
from config import Config
from loader import get_dataloader
from evaluator import Evaluator
import os
model = BertCNNHeavy(Config)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model.load_state_dict(torch.load(model_path))
    # model.to(device)
    return model

def test_model(model,config):
    test_loader = get_dataloader(config,split="test")
    evaluator = Evaluator(model,config)
    results = evaluator.eval(test_loader)
    print(results)

def save_onnx(model,config):
    batch_size = 1
    seq_len = config["max_length"]
    dummy_input = torch.zeros(batch_size,seq_len,dtype=torch.long)
    #dummy_input = dummy_input.to(device)
    model_path = os.path.join(config["model_path"],"model_%s.onnx"%(config["model_name"]))
    torch.onnx.export(model,dummy_input,
                      model_path,
                      input_names=["input_ids"],
                      output_names=["labels"],
                      opset_version=11
                    )
    print("save onnx model to %s"%(model_path))

def print_model_device_allocation(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Device: {param.device}")



def main(config):
    model_path = os.path.join(config["model_path"],"model_%s.pt"%(config["model_name"]))
    model = load_model(model_path)
    # print_model_device_allocation(model)
    # test_model(model,config)
    # save as onnx
    save_onnx(model,config)


if __name__ == '__main__':
    main(Config)
