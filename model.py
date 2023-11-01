from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
from transformers import BertModel


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

class TorchModel(nn.Module):
    def __init__(self,config):
        super(TorchModel, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        if config["model_with_bert"] :
            self.hidden_size = self.bert.config.hidden_size
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        self.model_type = config["model_type"]
        self.classify = nn.Linear(self.hidden_size, self.num_classes)
        # for not using pretrain model
        # self.embedding = nn.Embedding(config["vocab_size"], self.hidden_size)

    def forward(self,x):
        '''
        :param x: batch_size, seq_len
        :return: batch_size, num_classes
        '''
        x = self.bert(x) # (batch_size, seq_len, hidden_size)
        if isinstance(x, tuple):
            x = x[0]
        elif not isinstance(x, torch.Tensor):
            try:
                x = x.last_hidden_state  # for bert
            except:
                raise ValueError('x must be tuple or tensor')

        # get last hidden state
        x = x[:, -1, :]  # (batch_size, hidden_size)
        y_pred = self.classify(x) # (batch_size, num_classes)
        return y_pred



