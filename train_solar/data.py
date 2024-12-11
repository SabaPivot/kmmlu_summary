from datasets import load_dataset

def transform_and_format(data, tokenizer, if_train: bool):
    """
    Mapping function returns formatted 'text'
    """
    prompt = f"{data['question']}\n" \
             f"A: {data['A']}\n" \
             f"B: {data['B']}\n" \
             f"C: {data['C']}\n" \
             f"D: {data['D']}"

    if if_train:
        answer_map = {1: "A", 2: "B", 3: "C", 4: "D"}
        answer_choice = answer_map.get(data['answer'], "")
        
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
        conversations = [
            {"role": "user", "content": prompt},
        ]
        
        text = tokenizer.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        return {"text": text}


def load_train_data_in_chat_template(data_path, domain, tokenizer):
    data = load_dataset(data_path, domain)

    train_data = data["train"].map(lambda x: transform_and_format(x, tokenizer, True))

    return train_data

def load_test_data_in_chat_template(data_path, domain, tokenizer):
    data = load_dataset(data_path, domain)

    dev_data = data["dev"].map(lambda x: transform_and_format(x, tokenizer, False))
    test_data = data["test"].map(lambda x: transform_and_format(x, tokenizer, False))
    
    return dev_data, test_data