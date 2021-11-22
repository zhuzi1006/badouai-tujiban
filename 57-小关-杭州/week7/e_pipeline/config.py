"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "./data/train_tag_news.json",
    "valid_data_path": "./data/valid_tag_news.json",
    "test_data_path": "./data/tag_news.json",
    "vocab_path": "./data/chars.txt",
    "model_type": "cnn",
    "max_length": 20,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\_Gx\CodingSquare\badou\pretrain_models\bert-base-chinese",
    "seed": 1024,
}