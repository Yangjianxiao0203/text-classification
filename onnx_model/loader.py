import torch
from datasets import load_dataset,load_from_disk
from config import Config
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def download_dataset(path):
    if os.path.exists(path):
        return load_from_disk(path)
    logger.info("Downloading dataset")
    dataset = load_dataset("dair-ai/emotion")
    dataset.save_to_disk(path)
    return dataset

class DataGenerator(Dataset):
    def __init__(self,config,split = 'train'):
        super(DataGenerator,self).__init__()
        self.config = config
        self.data = download_dataset(config["data_path"])
        self.data = self.data[split]
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])
        self.load()

    def load(self):
        # encode all texts
        self.data_encode = self.data.map(self.encode)


    def encode(self,example):
        text = example["text"]
        label = example["label"]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config["max_length"],
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_attention_mask=False,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].flatten().clone().detach()
        return {
            "input_ids":input_ids,
            "label":label
        }

    def __len__(self):
        return len(self.data_encode)

    def __getitem__(self,idx):
        data = self.data_encode[idx]
        # 一定要在这里转化类型，否则跑不通
        x = torch.tensor(data["input_ids"],dtype=torch.long)
        y = torch.tensor(data["label"],dtype=torch.long)
        return x,y

def get_dataloader(config,split = 'train'):
    dataset = DataGenerator(config,split)
    dataloader = DataLoader(dataset,batch_size=config["batch_size"],shuffle=True)
    return dataloader

if __name__ == "__main__":
    config = Config
    dataset = DataGenerator(config,'validation')
    print(len(dataset))
    dataloader = DataLoader(dataset,batch_size=config["batch_size"],shuffle=True)
    for x,y in dataloader:
        print(x.shape)
        print(y.shape)