from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
from transformers import BertModel,BertConfig
import json

def load_bert(pretrain_path, config_path=None):
    if config_path is None:
        model = BertModel.from_pretrained(pretrain_path)
        return model
    with open(config_path) as f:
        config_dict = json.load(f)
    custom_config = BertConfig(**config_dict)
    model = BertModel.from_pretrained(pretrain_path, config=custom_config)
    return model
    
def get_bert_last_hidden_state(x):
    '''
    :return: batch_size, seq_len, hidden_size
    '''
    if isinstance(x, tuple):
        x = x[0]
    elif not isinstance(x, torch.Tensor):
        try:
            x = x.last_hidden_state  # for bert
        except:
            raise ValueError('x must be tuple or tensor')
    return x
class BayesModel:
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, dataloader):
        all_bows = []
        all_labels = []
        for x,y in dataloader:
            all_bows.extend(x.numpy())
            all_labels.extend(y.numpy())

        self.model.fit(all_bows, all_labels)

    def predict(self, bow):
        return self.model.predict(bow)

class Bert(nn.Module):
    def __init__(self,config):
        super(Bert, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        # self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        self.bert_embedding = load_bert(config["pretrain_model_path"], config["bert_config"])
        if config["model_with_bert"] :
            self.hidden_size = self.bert_embedding.config.hidden_size
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        self.classify = nn.Linear(self.hidden_size, self.num_classes)
        # for not using pretrain model
        # self.embedding = nn.Embedding(config["vocab_size"], self.hidden_size)

    def forward(self,x):
        '''
        :param x: batch_size, seq_len
        :return: batch_size, num_classes
        '''
        x = self.bert_embedding(x) # (batch_size, seq_len, hidden_size)
        x = get_bert_last_hidden_state(x)
        # get last hidden state
        x = x[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(x) # (batch_size, num_classes)
        return y_pred

class BertCNNModel(nn.Module):
    def __init__(self,config):
        super(BertCNNModel, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        self.seq_len = config["max_length"]
        # with bert embedding
        self.bert_embedding = load_bert(config["pretrain_model_path"], config["bert_config"]) # batch x seq_len x bert_embedding.config.hidden_size ->768
        if config["model_with_bert"] :
            self.hidden_size = self.bert_embedding.config.hidden_size
        self.kernel_size = config["kernel_size"]
        pad = int((self.kernel_size - 1)/2)
        #input channel = hidden_size, output channel = hidden_size
        self.conv = nn.Conv1d(self.hidden_size, self.hidden_size, self.kernel_size, padding=pad,bias=False)
        self.classify = nn.Linear(self.hidden_size , self.num_classes)

    def forward(self,x):
        '''
        :param x: batch_size, seq_len
        :return: batch_size, num_classes
        '''
        x = self.bert_embedding(x) # (batch_size, seq_len, hidden_size)
        x= get_bert_last_hidden_state(x)
        x = x.transpose(1,2) # (batch_size, hidden_size, seq_len) -> batch x channel x seq_len
        x = self.conv(x) # (batch_size, hidden_size, seq_len)
        x = x.transpose(1,2) # (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(x) # (batch_size, num_classes)
        return y_pred

class BertCNNHeavyModel(BertCNNModel):
    def __init__(self,config):
        '''
        add dropout, multi conv layer, batch norm, and activation function
        '''
        super(BertCNNHeavyModel, self).__init__(config)
        self.dropout = nn.Dropout(config["dropout"])
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size,eps=1e-05)
        self.convs = nn.ModuleList([])
        self.pooling = nn.MaxPool1d(self.seq_len)
        for _ in range(self.num_layers):
            self.convs.append(self.conv)
            self.convs.append(self.batch_norm)
            self.convs.append(self.activation)
            self.convs.append(self.dropout)

    def forward(self,x):
        '''
        :param x: batch_size, seq_len
        :return: batch_size, num_classes
        '''
        x = self.bert_embedding(x)
        x= get_bert_last_hidden_state(x)
        x = x.transpose(1,2) # (batch_size, hidden_size, seq_len) -> batch x channel x seq_len
        for conv in self.convs:
            x = conv(x)
        x = self.pooling(x).squeeze(-1) # (batch_size, hidden_size), last layer pooling: seq_len -> 1
        y_pred = self.classify(x) # (batch_size, num_classes)
        return y_pred


class BertLstmModel(nn.Module):
    #TODO:  num_layers 放在lstm中，看看要不要加dropout
    def __init__(self,config):
        super(BertLstmModel, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        # with bert embedding
        self.bert_embedding = load_bert(config["pretrain_model_path"], config["bert_config"])
        if config["model_with_bert"] :
            self.hidden_size = self.bert_embedding.config.hidden_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.classify = nn.Linear(self.hidden_size , self.num_classes)

    def forward(self,x):
        '''
        :param x:  batch_size, seq_len
        :return:
        '''
        x = self.bert_embedding(x)
        x = get_bert_last_hidden_state(x) # (batch_size, seq_len, hidden_size)
        x, _ = self.lstm(x) # (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(x) # (batch_size, num_classes)
        return y_pred
