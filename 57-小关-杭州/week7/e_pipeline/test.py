from loader import load_data
from model import TorchModel
from config import Config
import os
import torch


def load_model(config):
    model = TorchModel(config)
    parameters_path = os.path.join(config["model_path"], (config["model_type"] + '_lr' + str(config["learning_rate"])))
    parameters = torch.load(parameters_path)
    model.load_state_dict(parameters)

    return model


def main(config):
    correct, wrong = 0, 0
    test_data = load_data(config["test_data_path"], config)
    for model in ["gated_cnn", "cnn", "lstm"]:
        config["model_type"] = model
        for lr in [1e-2, 1e-3]:
            config["learning_rate"] = lr
            model = load_model(config)
            model.eval()
            for index, batch_data in enumerate(test_data):
                if torch.cuda.is_available():
                    batch_data = [b.cuda() for b in batch_data]
                input_ids, labels = batch_data
                with torch.no_grad():
                    pred_results = model(input_ids)
                for true_label, pred_label in zip(labels, pred_results):
                    pred_label = torch.argmax(pred_label)
                    if int(true_label) == int(pred_label):
                        correct += 1
                    else:
                        wrong += 1
            print('-' * 30)
            print(('模型' + config["model_type"] + '_lr' + str(config["learning_rate"]) + '准确率为：%.7f') % (correct / (correct + wrong)))


if __name__ == "__main__":
    main(Config)

    """
    output:
    ------------------------------
    模型gated_cnn_lr0.01准确率为：0.9366667
    ------------------------------
    模型gated_cnn_lr0.001准确率为：0.9366667
    ------------------------------
    模型cnn_lr0.01准确率为：0.9346296
    ------------------------------
    模型cnn_lr0.001准确率为：0.9344444
    ------------------------------
    模型lstm_lr0.01准确率为：0.9338889
    ------------------------------
    模型lstm_lr0.001准确率为：0.9194444

    """

