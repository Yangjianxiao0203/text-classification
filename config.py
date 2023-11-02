Config = {
    "train_dir":"./data/train",
    "test_dir":"./data/test",
    "valid_dir":"./data/validation",
    "model_path":"./output",
    "eval_path":"./eval",
    "pretrain_model_path": r"bert-base-uncased",
    "bert_config": "bert_config.json",

    "encoding":"bert", #bayes will be bow
    "model_type":"bert_cnn",
    "model_with_bert":True,

    "epoch": 8,
    "num_layers": 1,
    "max_length": 128,
    "hidden_size": 128,
    "num_classes":6,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "seed":768,

    #for only cnn related
    "kernel_size": 3,

    "optimizer": "adam",
    "loss_fn": "cross_entropy",

    "debug_mode":True,

}