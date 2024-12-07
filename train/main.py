import os, yaml
import torch
from model import load_peft_model
from data import load_data_in_chat_template
from train import set_trainer

os.environ["WANDB_PROJECT"] = "KMMLU"
with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    # Load model with Unsloth
    model_name = config["model"]["model_path"]
    model, tokenizer = load_peft_model(model_name)

    # Load Datasets
    data_path, domain = "HAERAE-HUB/KMMLU", config["dataset"]["domain"]
    train_data, dev_data, test_data = load_data_in_chat_template(data_path, domain, tokenizer)

    # Start Train
    trainer = set_trainer(model, tokenizer, train_data)
    trainer.train()