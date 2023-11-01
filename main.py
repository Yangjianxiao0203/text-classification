import os
import random

import numpy as np
import torch
from config import Config
from evaluator import ML_Evaluator,BertEvaluator
from loader import get_dataloader
from model import *
from optimizer import choose_optimizer,choose_loss
from utils.save_functions import save_as_json
from torch.utils.tensorboard import SummaryWriter

import logging
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
writer = SummaryWriter()

def train_by_bayes(save_model=True,save_eval=True):
    train_data,vector = get_dataloader()
    model = BayesModel()
    model.fit(train_data)

    # evaluate
    evaluator = ML_Evaluator(model, None)
    test_loader,_ = get_dataloader(train=False, vectorizer=vector)
    result_json = evaluator.evaluate(test_loader)
    if save_eval:
        save_path = Config["eval_path"]
        save_file = "bayes_eval.json"
        save_as_json(save_path,save_file,result_json)
    return

def choose_model(config):
    if config["model_type"] == 'bert':
        return Bert(config)
    elif config["model_type"] == 'bert_cnn':
        return BertCNNModel(config)
    else:
        raise ValueError("model type not supported")

def train_by_nn(config,verbose=True):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data,_ = get_dataloader()
    model = choose_model(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu is available, move model to gpu")
        model = model.cuda()
    optimizer = choose_optimizer(config,model)
    loss_fn = choose_loss(config)
    evaluator = BertEvaluator(model)
    valid_data,_ = get_dataloader(valid = True)
    # train
    for epoch in range(config["epoch"]):
        model.train()

        if verbose:
            logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            x, y = batch_data
            y_pred = model(x)
            loss = loss_fn(y_pred, y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if verbose and index % 100 == 0:
                logger.info("epoch %d batch %d loss %.4f" % (epoch, index, loss.item()))

        if verbose:
            logger.info("epoch %d loss %.4f" % (epoch, np.mean(train_loss)))
            writer.add_scalar('train_loss', np.mean(train_loss), epoch)
        # evaluate
        acc = evaluator.evaluate(valid_data)
        writer.add_scalar('valid_acc', acc, epoch)
        if verbose:
            logger.info("epoch %d acc %.4f" % (epoch, acc))

    return acc



def train(Config):
    if Config["model_type"] == "bayes":
        train_by_bayes()
    else:
        train_by_nn(Config)

if __name__=='__main__':
    train(Config)