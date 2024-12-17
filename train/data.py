import yaml
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)

data = load_dataset(config["dataset"]["dataset_path"], config["dataset"]["domain"])


def get_unsloth_tokenizer(tokenizer):
    """
    Get Unsloth tokenier, to execute chat_template method
    """
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5"
    )
    return tokenizer


def get_prompt(data):
    prompt = f"{data['question']}\n" \
             f"A: {data['A']}\n" \
             f"B: {data['B']}\n" \
             f"C: {data['C']}\n" \
             f"D: {data['D']}"
    return prompt


def get_answer_choice(answer_key):
    answer_map = {1: "A", 2: "B", 3: "C", 4: "D"}
    return answer_map.get(answer_key, "")


def transform_and_format(data, tokenizer, train: bool, fewshot: bool, cot: bool):
    """
    Mapping function returns QWEN 2.5 formatted 'text'
    """
    prompt = get_prompt(data)

    if train:
        answer_choice = get_answer_choice(data['answer'])
        
        # Construct the conversations
        conversations = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_choice}
        ]

        # Apply the chat template
        text = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        
        return {"text": text}

    else:
        if cot:  # Chain of Thought Few-shot
    conversations = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        *load_chain_of_thought_in_chat_template(),
        {"role": "user", "content": prompt},
    ]
    
    elif fewshot:  # Few-shot (without CoT)
        conversations = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            *load_fewshot_data_in_chat_template(),
            {"role": "user", "content": prompt},
        ]
    else:  # Zero-shot
        conversations = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Answer with one alphabet letter."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        return {"text": text}


def load_train_data_in_chat_template(tokenizer):
    """
    Get model tokenizer and return train data with chat template applied.
    """
    tokenizer = get_unsloth_tokenizer(tokenizer)

    # always set train=True
    train_data = data["train"].map(lambda x: transform_and_format(x, tokenizer, train=True, fewshot=False))

    return train_data


def load_test_data_in_chat_template(tokenizer, fewshot, cot):
    """
    Get model tokenizer and return test data with chat template applied.

    You can give direct fewshot using dev dataset, with giving arguemnt fewshot=True in "transfrom_and_format" function
    """
    tokenizer = get_unsloth_tokenizer(tokenizer)
    
    # To apply few shot set fewshot=True
    test_data = data["test"].map(lambda x: transform_and_format(x, tokenizer, train=False, fewshot=fewshot, cot=cot))
    
    return test_data


def load_fewshot_data_in_chat_template():
    """
    load 5 fewshot data from "dev" data, which contains 5 rows data.
    """
    dev_data = data["dev"]
    
    few_shot_conversations = []
    for i in range(4):
        question = get_prompt(dev_data[i])
        answer_choice = get_answer_choice(dev_data[i]['answer'])
        
        few_shot_conversations.append({"role": "user", "content": question})
        few_shot_conversations.append({"role": "assistant", "content": answer_choice})

    return few_shot_conversations


def load_chain_of_thought_in_chat_template():
    """
    load 5 fewshot data from "dev" split "chain_of_thought" column, which contains 5 rows data.
    """
    dev_data = data["dev"]
    
    chain_of_thought_conversations = []
    for i in range(len(dev_data)):
        question = get_prompt(dev_data[i])
        chain_of_thought = dev_data[i]["chain_of_thought"]
        
        few_shot_conversations.append({"role": "user", "content": question})
        few_shot_conversations.append({"role": "assistant", "content": chain_of_thought})

    return chain_of_thought_conversations
