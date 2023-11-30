import torch
import torch.nn as nn
from transformers import BertModel,BertConfig
from config import Config

class BertCNNHeavy(nn.Module):
    def __init__(self,config):
        super(BertCNNHeavy, self).__init__()
        self.config = config
        # self.encoder = BertModel.from_pretrained(config["pretrain_model_path"])
        self.encoder = self.get_encoder(config)
        self.num_classes = config["num_classes"]
        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = config["num_layers"]
        self.dropout = nn.Dropout(config["dropout"])
        self.conv_stack = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.conv_stack.append(nn.Conv1d(self.hidden_size,self.hidden_size,3,padding=1))
            self.conv_stack.append(nn.ReLU())
            self.conv_stack.append(nn.BatchNorm1d(self.hidden_size,eps=1e-05))
            self.conv_stack.append(self.dropout)
        self.classify = nn.Linear(self.hidden_size,self.num_classes)

    def get_encoder(self,config):
        encoder = BertModel.from_pretrained(config["pretrained_model_name"])
        config_dict = encoder.config.to_dict()
        config_dict["num_hidden_layers"] = config["bert_layers"]
        encoder_config = BertConfig(**config_dict)
        encoder = BertModel.from_pretrained(config["pretrained_model_name"],config=encoder_config)
        return encoder


    def forward(self,x):
        x = self.encoder(x)
        x= x.last_hidden_state # batch x seq_len x hidden_size
        x = x.transpose(1,2) # batch x hidden_size x seq_len
        for conv in self.conv_stack:
            x = conv(x)
        pooling = nn.MaxPool1d(x.size(-1))
        x = pooling(x).squeeze(-1) # batch x hidden_size
        y_pred = self.classify(x) # batch x num_classes
        return y_pred


if __name__ == '__main__':
    model = BertCNNHeavy(Config)
    # 查看所有模型层的名字
    for name, param in model.named_parameters():
        # 查看所有参与训练的参数
        if param.requires_grad:
            print(name)