from config import Config
from model import BertCNNHeavy
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loader import get_dataloader
from evaluator import Evaluator
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

def train(config,verbose=True,save_model=False):
    epochs = config["epoch"]
    if Debug:
        epochs = 1
    model = BertCNNHeavy(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    evaluator = Evaluator(model,config)
    train_loader = get_dataloader(config,split="train")
    valid_loader = get_dataloader(config,split="validation")
    test_loader = get_dataloader(config,split="test")
    logger.info("start training")
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for idx,(x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if verbose and idx % 10 == 0:
                logger.info("epoch %d, batch %d, loss %f" % (epoch,idx,np.mean(train_loss)))
            if Debug:
                break
        writer.add_scalar("train_loss",np.mean(train_loss),epoch)
        results = evaluator.eval(valid_loader)
        logger.info("epoch %d, train loss %f, valid acc %f" % (epoch,np.mean(train_loss),results["accuracy"]))

    if save_model:
        path = config["model_path"]
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(model.state_dict(),os.path.join(path,"model_%s.pt"%(config["model_name"])))

    return model


if __name__ == "__main__":
    train(Config,verbose=True,save_model=True)