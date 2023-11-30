from model import BertCNNHeavy
import torch
from config import Config
from loader import get_dataloader
from evaluator import Evaluator
import os
model = BertCNNHeavy(Config)

def load_model(model_path):
    model.load_state_dict(torch.load(model_path))
    return model

def test_model(model,config):
    test_loader = get_dataloader(config,split="test")
    evaluator = Evaluator(model,config)
    results = evaluator.eval(test_loader)
    print(results)

def main(config):
    model_path = os.path.join(config["model_path"],"model_%s.pt"%(config["model_name"]))
    model = load_model(model_path)
    test_model(model,config)


if __name__ == '__main__':
    main(Config)
