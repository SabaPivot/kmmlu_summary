import os, yaml, argparse
import torch
from model import load_peft_model
from data import load_train_data_in_chat_template, load_test_data_in_chat_template
from train import set_trainer
from inference import inference

os.environ["WANDB_PROJECT"] = "KMMLU"
with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)

def main(mode):
    # Load model with Unsloth
    model_name = config["model"]["model_path"]
    model, tokenizer = load_peft_model(model_name)

    # Load Datasets
    data_path, domain = "HAERAE-HUB/KMMLU", config["dataset"]["domain"]

    # Load data in chat template
    train_data = load_train_data_in_chat_template(data_path, domain, tokenizer)
    dev_data, test_data = load_test_data_in_chat_template(data_path, domain, tokenizer)

    if mode == "train":
        trainer = set_trainer(model, tokenizer, train_data)
        trainer.train()
    
    elif mode == "inference":
        inference(model, tokenizer, dev_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMMLU Train/Inference Script")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Mode to run: 'train' or 'inference'")
    args = parser.parse_args()

    main(args.mode)
