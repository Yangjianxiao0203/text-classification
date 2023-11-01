Config = {
    "train_dir":"./data/train",
    "test_dir":"./data/test",
    "valid_dir":"./data/validation",
    "model_path":"./output",
    "eval_path":"./eval",
    "pretrain_model_path": r"bert-base-uncased",

    "encoding":"bert", #bayes will be bow
    "model_type":"bert_cnn",
    "model_with_bert":True,

    "epoch": 10,
    "num_layers": 2,
    "max_length": 256,
    "hidden_size": 128,
    "num_classes":6,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "seed":768,

    #for only cnn related
    "kernel_size": 3,

    "optimizer": "adam",
    "loss_fn": "cross_entropy",

}