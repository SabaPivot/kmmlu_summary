import torch
from tqdm import tqdm

def inference(model, tokenizer, data):
    data, kmmlu_ans = data["text"], data["answer"]

    answers = []
    for inputs in tqdm(data):
        inputs = torch.tensor(inputs).to('cuda')
        input_length = inputs.shape[-1]
        outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=1, 
        do_sample=False,
        # use_cache=True, 
        temperature=0.05,
        min_p=0.1
        )

        outputs = outputs[:, input_length:]
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers.append(result)

    # Evaluation
    answers = [1 if x == 'A' else 2 if x == 'B' else 3 if x == 'C' else 4 if x == 'D' else x for sublist in answers for x in sublist]
    count = 0
    for i in range(len(answers)):
        print(answers[i], kmmlu_ans[i])
        if answers[i] == kmmlu_ans[i]:
            count += 1
    print(f"{count}/{len(answers)}")
    
    return answers