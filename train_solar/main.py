import os, yaml, argparse
import torch
from model import load_peft_model, load_model
from data import load_train_data_in_chat_template, load_test_data_in_chat_template
from train import set_trainer
from inference import inference

os.environ["WANDB_PROJECT"] = "KMMLU_solar"
with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)

def main(mode):
    # Load model with Unsloth
    model_name = config["model"]["model_path"]
    data_path, domain = config["dataset"]["dataset_path"], config["dataset"]["domain"]

    if mode == "train":
        # Load model and tokenizer
        model, tokenizer = load_peft_model(model_name)

        # Load data
        train_data = load_train_data_in_chat_template(data_path, domain, tokenizer)

        # Start Training
        trainer = set_trainer(model, tokenizer, train_data)
        trainer.train(resume_from_checkpoint=True)
    
    elif mode == "inference":
        model, tokenizer = load_model(model_name)
        dev_data, test_data = load_test_data_in_chat_template(data_path, domain, tokenizer)
        inference(model, tokenizer, test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMMLU Train/Inference Script")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Mode to run: 'train' or 'inference'")
    args = parser.parse_args()

    main(args.mode)
