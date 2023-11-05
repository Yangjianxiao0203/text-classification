from peftModel import get_lora_model
import os
import random

import numpy as np
import torch
from config import Config
from evaluator import ML_Evaluator,BertEvaluator
from loader import get_dataloader
from model import *
from optimizer import choose_optimizer,choose_loss
from utils.save_functions import save_as_json, save_results_to_json, save_all_json_to_csv,save_results_to_csv
from torch.utils.tensorboard import SummaryWriter
from utils.logging import log_and_write_metrics

import logging
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
writer = SummaryWriter()

Debug = Config['debug_mode']

def train(config):
    model = get_lora_model(config)
    #print trainnable parameters
    print("trainable parameters:"+"*"*20)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("*"*20)
    print(model.print_trainable_parameters())
    train_data, _ = get_dataloader(config=config)
    valid_data,_ = get_dataloader(valid = True,config=config)
    test_data, _ = get_dataloader(train=False, config=config)
    epochs = config["epoch"]
    optimizer = choose_optimizer(config,model)
    loss_fn = choose_loss(config)
    cuda_flag = torch.cuda.is_available()
    evaluator = BertEvaluator(model)
    if cuda_flag:
        logger.info("gpu is available, move model to gpu")
        model = model.cuda()

    if Debug:
        epochs = 1

    for epoch in range(epochs):
        model.train()

        logger.info("epoch: {}".format(epoch))
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [data.cuda() for data in batch_data]
            x, y = batch_data
            optimizer.zero_grad()
            y_pred = model(x)
            if not isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.logits
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if index % 100 == 0:
                logger.info("epoch: {}, index: {}, loss: {}".format(epoch, index, loss.item()))
            if Debug:
                break
        results = evaluator.evaluate(valid_data)
        log_and_write_metrics(writer, logger, results, epoch, True, data_type="valid")

    results = evaluator.evaluate(test_data)
    log_and_write_metrics(writer, logger, results, epochs, True, data_type="test")
    return results


if __name__ =='__main__':
    train(Config)
