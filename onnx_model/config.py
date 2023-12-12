Config = {
    "data_path":"../data",
    "model_path":"./model",
    "model_name":"bert_cnn_heavy",
    "pretrained_model_name":"bert-base-uncased",
    "num_classes": 6,
    "bert_layers":2,
    "num_layers": 3,
    "max_length": 128,
    "batch_size": 64,
    "lr": 1e-4,
    "dropout": 0.1,
    "optimizer": "adam",
    "epoch": 1,

    "seed": 42,
    "debug_mode":False,
}