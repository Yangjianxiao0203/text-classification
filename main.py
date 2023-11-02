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

#TODO: 1.增加模型，2.让num layers有用，加入dropout 3. 用LoRA微调bert 4. 用自己的embedding
def train_by_bayes(config,save_model=True,save_eval=True):
    train_data,vector = get_dataloader(config)
    model = BayesModel()
    model.fit(train_data)

    # evaluate
    evaluator = ML_Evaluator(model, None)
    test_loader,_ = get_dataloader(config,train=False, vectorizer=vector)
    result_json = evaluator.evaluate(test_loader)
    if save_eval:
        save_path = config["eval_path"]
        save_file = "bayes_eval.json"
        save_as_json(save_path,save_file,result_json)
    return

def train_by_nn(config,verbose=True,save_model=False):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data, _ = get_dataloader(config=config)
    model = choose_model(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu is available, move model to gpu")
        model = model.cuda()
    optimizer = choose_optimizer(config,model)
    loss_fn = choose_loss(config)
    evaluator = BertEvaluator(model)
    valid_data,_ = get_dataloader(valid = True,config=config)
    test_data, _ = get_dataloader(train=False, config=config)

    epochs = config["epoch"]
    if Debug:
        epochs = 1
    # train
    for epoch in range(epochs):
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

            if verbose and index % 50 == 0:
                logger.info("epoch %d batch %d loss %.4f" % (epoch, index, loss.item()))
            if Debug:
                break

        if verbose:
            logger.info("epoch %d loss %.4f" % (epoch, np.mean(train_loss)))
            writer.add_scalar('train_loss', np.mean(train_loss), epoch)
        results = evaluator.evaluate(valid_data)
        log_and_write_metrics(writer, logger, results, epoch, verbose,data_type="valid")

    save_file_header = config["model_type"]+"_b_"+str(config["batch_size"]) + "_lr_" + str(Config["learning_rate"])
    if save_model:
        save_path = config["model_path"]
        model_save_path = os.path.join(save_path, save_file_header + "_epoch_" + str(epoch) + ".pt")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")

    # evaluate on test data
    results = evaluator.evaluate(test_data)
    log_and_write_metrics(writer, logger, results, epoch, verbose,data_type="test")

    return results


def choose_model(config):
    if config["model_type"] == 'bert':
        return Bert(config)
    elif config["model_type"] == 'bert_cnn':
        return BertCNNModel(config)
    elif config["model_type"] == 'bert_cnn_heavy':
        return BertCNNHeavyModel(config)
    elif config["model_type"] == 'bert_lstm':
        return BertLstmModel(config)
    else:
        raise ValueError("model type not supported")

def train(Config):
    if Config["model_type"] == "bayes":
        return train_by_bayes(Config)
    else:
        return train_by_nn(Config)

if __name__=='__main__':
    # train(Config)
    
    # grid search
    models = ["bert_cnn_heavy","bert","bert_lstm"]
    batch_sizes = [64, 32]
    learning_rates = [1e-4]
    max_lengths = [64]
    num_layers = [1,2]

    for model in models:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for max_length in max_lengths:
                    for num_layer in num_layers:
                        Config["num_layers"] = num_layer
                        Config["model_type"] = model
                        Config["batch_size"] = batch_size
                        Config["learning_rate"] = learning_rate
                        Config["max_length"] = max_length
                        logger.info("****************start training**************")
                        logger.info(f"model_type: {model}, batch_size: {batch_size}, learning_rate: {learning_rate}, max_length: {max_length}, num_layer: {num_layer}")
                        results = train_by_nn(Config)
                        if Debug:
                            break
                        save_results_to_json(results, Config)
                        save_results_to_csv(results, Config)
                        logger.info("save file to json and csv")

    logger.info("****************finish training**************")
    