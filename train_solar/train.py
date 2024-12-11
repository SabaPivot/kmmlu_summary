from transformers import TrainingArguments
from trl import SFTTrainer
import yaml
import torch


with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)
config["training_args"]["bf16"] = torch.cuda.is_bf16_supported()
config["training_args"]["fp16"] = not torch.cuda.is_bf16_supported()

def set_trainer(model, tokenizer, train_data):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_data,
        dataset_text_field = "text",
        max_seq_length = config["dataset"]["max_seq_length"],
        dataset_num_proc=config["dataset"]["dataset_num_proc"],
        packing=config["dataset"]["packing"],
        args=TrainingArguments(**config["training_args"]),
        )
    return trainer