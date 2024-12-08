from unsloth import FastLanguageModel
import torch


def inference(model, tokenizer, data):
    FastLanguageModel.for_inference(model)
    data = data["text"]

    print(tokenizer.batch_decode(data[0]))

    answers = []
    for inputs in data:
        inputs = torch.tensor(inputs).to('cuda')
        input_length = inputs.shape[-1]
        outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128, 
        use_cache=True, 
        temperature=0.7,
        top_p=0.9,
        min_p=0.1
        )

        outputs = outputs[:, input_length:]
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers.append(result)

    for ans in answers:
        print(ans)
    
    return answers