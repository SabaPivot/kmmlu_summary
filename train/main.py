import os
import torch
from model import load_peft_model
from data import load_data_in_chat_template
from train import set_trainer

os.environ["WANDB_PROJECT"] = "KMMLU"

if __name__ == "__main__":
    # Load model with Unsloth
    model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"
    model, tokenizer = load_peft_model(model_name)

    # Load Datasets
    data_path, domain = "HAERAE-HUB/KMMLU", "Biology"
    train_data, dev_data, test_data = load_data_in_chat_template(data_path, domain, tokenizer)

    # Start Train
    trainer = set_trainer(model, tokenizer, train_data)
    trainer.train()