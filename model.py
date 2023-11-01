from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
from transformers import BertModel

def get_bert_last_hidden_state(x):
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

class BertModel(nn.Module):
    def __init__(self,config):
        super(BertModel, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        if config["model_with_bert"] :
            self.hidden_size = self.bert.config.hidden_size
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
        x = self.bert(x) # (batch_size, seq_len, hidden_size)
        x = get_bert_last_hidden_state(x)
        # if isinstance(x, tuple):
        #     x = x[0]
        # elif not isinstance(x, torch.Tensor):
        #     try:
        #         x = x.last_hidden_state  # for bert
        #     except:
        #         raise ValueError('x must be tuple or tensor')

        # get last hidden state
        x = x[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(x) # (batch_size, num_classes)
        return y_pred

class CNNModel(nn.Module):
    def __init__(self,config):
        super(CNNModel, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        # with bert embedding
        self.bert_embedding = BertModel.from_pretrained(config["pretrain_model_path"]) # batch x seq_len x bert_embedding.config.hidden_size ->768
        if config["model_with_bert"] :
            self.hidden_size = self.bert_embedding.config.hidden_size
        self.kernel_size = config["kernel_size"]
        pad = int((self.kernel_size - 1)/2)
        self.conv = nn.Conv1d(self.hidden_size, self.hidden_size, self.kernel_size, padding=pad,bias=False)
        self.classify = nn.Linear(self.hidden_size * 3, self.num_classes)

    def forward(self,x):
        '''
        :param x: batch_size, seq_len
        :return: batch_size, num_classes
        '''
        x = self.bert_embedding(x) # (batch_size, seq_len, hidden_size)
        x= get_bert_last_hidden_state(x)




