from langchain_upstage import ChatUpstage
import os
import time


def four_to_choice(data) -> dict[str, list]:
    """
    A, B, C, D 선택지를 하나의 choices(list)로 합쳐서 넣어주는 함수
    """
    choices = [data['A'], data['B'], data['C'], data['D']]
    return {'choices': choices}

def get_answer_choice(answer_key):
    answer_map = {1: "A", 2: "B", 3: "C", 4: "D"}
    return answer_map.get(answer_key, "")


def load_model(model):
    return ChatUpstage(model=model, api_key=os.getenv("UPSTAGE_API_KEY"), temperature=0.0)


def add_summarized_question(data, chain, num_try=0):
    try:
        summary = chain.invoke({"question": data['question']}).content
        if len(data['question']) >= len(summary) > len(data['question']) / 3:
            data['summarized_q'] = summary
        else:
            data['summarized_q'] = data['question']

        print(f"\n{data['question']}\nSummary: {data['summarized_q']}")
        return data
    except TimeoutError as e:
        print(num_try)
        if num_try > 100000:
            with open("output.txt", "a") as f:
                f.write(data['question'] + "\n")
        time.sleep(3)
        add_summarized_question(data, num_try+1)

def add_chain_of_thought(data, chain, num_try=0):
    try:
        chain_of_thought = chain.invoke({"question": data['question'], "choice": data['choices'], "answer": get_answer_choice(data["answer"])}).content
        print(chain_of_thought)
        data["chain_of_thought"] = chain_of_thought
        return data

    except TimeoutError as e:
        print(num_try)
        if num_try > 100000:
            with open("output.txt", "a") as f:
                f.write(data['question'] + "\n")
        time.sleep(3)
        add_summarized_question(data, num_try+1)
