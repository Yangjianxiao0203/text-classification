import json
from transformers import BertModel, BertConfig


def load_bert(pretrain_path, config_path=None):
    if config_path is None:
        model = BertModel.from_pretrained(pretrain_path)
        return model
    with open(config_path) as f:
        config_dict = json.load(f)
    custom_config = BertConfig(**config_dict)
    model = BertModel.from_pretrained(pretrain_path, config=custom_config)
    return model