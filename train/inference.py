from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

class AlphabetStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens, tokenizer):
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token_id = input_ids[0, -1].item()
        last_token = self.tokenizer.decode([last_token_id], skip_special_tokens=True).strip()

        return last_token in self.stop_tokens



def inference(model, tokenizer, data, fewshot, cot):
    FastLanguageModel.for_inference(model)
    data, kmmlu_ans = data["text"], data["answer"]

    if fewshot or cot:
        answers = []
        for inputs in tqdm(data):
            inputs = torch.tensor(inputs).to('cuda')
            input_length = inputs.shape[-1]
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.3,
            )

            outputs = outputs[:, input_length:] 

            result = ""
            for i in range(outputs.size(0)):
                decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
                if decoded[0] in {'A', 'B', 'C', 'D'}:
                    result = decoded[0]
                else:
                    for char in reversed(decoded):
                        if char in {'A', 'B', 'C', 'D'}:
                            result = char
                            break
                
                if result == "":
                    result = "A"
                    print(f"Add: {result}")
                    print(f"result: {decoded}")
                answers.append(result)

    else:
        stop_tokens = ["A", "B", "C", "D"]
        stopping_criteria = StoppingCriteriaList([AlphabetStoppingCriteria(stop_tokens, tokenizer)])

        answers = []
        for inputs in tqdm(data):
            inputs = torch.tensor(inputs).to('cuda')
            input_length = inputs.shape[-1]
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=2048,
                stopping_criteria=stopping_criteria,
                do_sample=False
            )

            outputs = outputs[:, input_length:]

            for i in range(outputs.size(0)):
                decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
                result = decoded.strip().split()[-1]
                
                if result not in stop_tokens:
                    print(f"Add: {result[-1]}")
                    print(f"result: {decoded}")
                answers.append(result[-1])

    answers = [
        1 if x in 'A' else
        2 if x in 'B' else
        3 if x in 'C' else
        4 if x in 'D' else x
        for x in answers
    ]

    count = 0
    for i in range(len(answers)):
        if answers[i] == kmmlu_ans[i]:
            count += 1
    print(f"{count}/{len(answers)}")
    
    return answers
