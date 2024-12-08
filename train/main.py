import os, yaml
import torch
from model import load_peft_model
from data import load_train_data_in_chat_template, load_test_data_in_chat_template
from train import set_trainer
from inference import inference

os.environ["WANDB_PROJECT"] = "KMMLU"
with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    # Load model with Unsloth
    model_name = config["model"]["model_path"]
    model, tokenizer = load_peft_model(model_name)

    # Load Datasets
    data_path, domain = "HAERAE-HUB/KMMLU", config["dataset"]["domain"]

    train_data = load_train_data_in_chat_template(data_path, domain, tokenizer)
    dev_data, train_data = load_test_data_in_chat_template(data_path, domain, tokenizer)

    # inference
    inference(model, dev_data)
    exit()

    # Start Train
    trainer = set_trainer(model, tokenizer, train_data)
    trainer.train()