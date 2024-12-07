from unsloth.chat_templates import get_chat_template
from datasets import load_dataset


def get_unsloth_tokenizer(tokenizer):
    """
    Get Unsloth tokenier, to execute chat_template method
    """
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5"
    )
    return tokenizer


def transform_and_format(data, tokenizer, if_train: bool):
    """
    Mapping function returns QWEN 2.5 formatted 'text'
    """
    prompt = f"{data['question']}\n" \
             f"A: {data['A']}\n" \
             f"B: {data['B']}\n" \
             f"C: {data['C']}\n" \
             f"D: {data['D']}"
    
    answer_map = {1: "A", 2: "B", 3: "C", 4: "D"}
    answer_choice = answer_map.get(data['answer'], "")

    if if_train:
        # Construct the conversations
        conversations = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_choice}
        ]

        # Apply the chat template
        text = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
    else:
        conversations = [
            {"role": "user", "content": prompt},
        ]
        
        text = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
    
    # Return the final output with conversations and text
    return {"text": text}


def load_data_in_chat_template(model_path, domain, tokenizer):
    data = load_dataset(model_path, domain)
    tokenizer = get_unsloth_tokenizer(tokenizer)
    train_data, dev_data, test_data = data["train"], data["dev"], data["test"]
    
    train_data = data["train"].map(lambda x: transform_and_format(x, tokenizer, True))
    dev_data = data["dev"].map(lambda x: transform_and_format(x, tokenizer, False))
    test_data = data["test"].map(lambda x: transform_and_format(x, tokenizer, False))
    
    return train_data, dev_data, test_data