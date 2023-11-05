from peft import get_peft_model,LoraConfig, TaskType
from utils.load_functions import load_bert
import torch
import tensorboard
import logging
from config import Config
from transformers import BertForSequenceClassification,BertConfig

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_lora_model(config):
    Bert = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=config["num_classes"],
        output_attentions=False,
        output_hidden_states=False,
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=True,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.1,
        target_modules=config["target_modules"]
    )

    model = get_peft_model(Bert, peft_config)
    #set this lora layer to be trainable
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

if __name__ == "__main__":
    # print out all parameters' name
    model = get_lora_model(Config)
    for k, v in model.named_parameters():
        print(k, v.shape)
    print("*"*20)
    print(model.print_trainable_parameters())