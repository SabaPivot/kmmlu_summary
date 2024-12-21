import os, yaml, argparse
import torch
from model import load_peft_model, load_model
from data import load_train_data_in_chat_template, load_test_data_in_chat_template
from train import set_trainer
from inference import inference

os.environ["WANDB_PROJECT"] = "KMMLU"
with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)

def main(mode, fewshot, cot):
    # Model name
    model_name = config["model"]["model_path"]

    if mode == "train":
        # Load model and tokenizer
        model, tokenizer = load_peft_model(model_name)

        # Load data
        train_data = load_train_data_in_chat_template(tokenizer)

        # Start Training
        trainer = set_trainer(model, tokenizer, train_data)
        trainer.train()
    
    elif mode == "inference":
        model, tokenizer = load_model(model_name)
        test_data = load_test_data_in_chat_template(tokenizer, fewshot=fewshot, cot=cot)
        test_data = test_data.shuffle(seed=42).select(range(100))
        inference(model, tokenizer, test_data, fewshot=fewshot, cot=cot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMMLU Train/Inference Script")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Mode to run: 'train' or 'inference'")
    parser.add_argument("--fewshot", default=False, action='store_true', help="if type ""--fewshot"", do fewshot with dev data, if nothing, 0-shot")
    parser.add_argument("--cot", default=False, action='store_true', help="if type ""--cot"", do chain_of_thought fewshot, if nothing, 0-shot")
    args = parser.parse_args()

    if args.fewshot and args.cot:
        raise ValueError("You cannot use fewshot and chain_of_thought at the same time.")

    main(args.mode, args.fewshot, args.cot)
