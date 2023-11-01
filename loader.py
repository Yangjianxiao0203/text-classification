import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset,set_caching_enabled,load_from_disk
from sklearn.feature_extraction.text import CountVectorizer

import logging
from config import Config

set_caching_enabled(False)
#set logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def download_dataset():
    #check if data directory exists
    if os.path.exists("./data"):
        return
    logger.info("Downloading dataset")
    dataset = load_dataset("dair-ai/emotion")
    #save in local
    dataset.save_to_disk("./data")

class DataGenerator(Dataset):
    def __init__(self,path):
        self.data = load_from_disk(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        label = sample["label"]
        return text,label

class BertVectorGenerator(Dataset):
    def __init__(self,config,path):
        self.data = load_from_disk(path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.config = config

    def get_vectorizer(self):
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        label = sample["label"]
        '''
        add_special_tokens: add [CLS] and [SEP] token
        max_length: max length of the sequence
        return_token_type_ids: return token type ids to indicate which token belongs to which sequence
        padding: pad to max_length
        return_tensors: return pytorch tensor
        '''
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config["max_length"],
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=False,
            return_tensors='pt',
        )
        x = encoding['input_ids'].flatten()
        y = torch.tensor(label, dtype=torch.long)
        return x, y

class BowGenerator(Dataset):
    def __init__(self,path,vectorizer=None):
        self.data = load_from_disk(path)
        self.texts = [item['text'] for item in self.data ]
        if vectorizer is None:
            self.vectorizer = CountVectorizer()
            self.bow_data = self.vectorizer.fit_transform(self.texts).toarray()
        else:
            self.vectorizer = vectorizer
            self.bow_data = self.vectorizer.transform(self.texts).toarray()

    def __len__(self):
        return len(self.data)

    def get_vectorizer(self):
        return self.vectorizer

    def __getitem__(self, idx):
        x = torch.tensor(self.bow_data[idx], dtype=torch.float32)
        y = torch.tensor(self.data[idx]['label'], dtype=torch.long)
        return x,y


def get_dataloader(config, shuffle=True,train=True,valid=False, vectorizer=None):
    path = config["train_dir"] if train else config["test_dir"]
    encoding = config["encoding"]
    batch_size = config["batch_size"]
    if valid:
        path = Config["valid_dir"]
    if encoding == "bert":
        dataset = BertVectorGenerator(config, path)
    elif encoding == "bow":
        dataset = BowGenerator(path,vectorizer)
    else:
        raise ValueError("encoding not supported")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, dataset.get_vectorizer()

if __name__ == '__main__':
    download_dataset()