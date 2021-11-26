import torch
import os
import random
import numpy as np
import logging
import pandas as pd
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from _collections import defaultdict

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu yes, model to gpu")
        model = model.cuda()
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], (config["model_type"] + '_lr' + str(config["learning_rate"])))
    torch.save(model.state_dict(), model_path)
    return acc


if __name__ == "__main__":
    result = defaultdict(list)
    for model in ["gated_cnn", "cnn", "lstm"]:
        Config["model_type"] = model
        for lr in [1e-2, 1e-3]:
            Config["learning_rate"] = lr
            acc = main(Config)
            Config.update({'acc': acc})
            for key, val in Config.items():
                result[key].append(val)
            print("最后一轮准确率：", main(Config), "当前配置：", Config)
    result = pd.DataFrame(result)
    result.to_csv('./output/result.csv')



