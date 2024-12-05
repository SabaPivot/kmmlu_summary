from langchain_upstage import ChatUpstage
import os
import time


def four_to_choice(data) -> dict[str, list]:
    """
    A, B, C, D 선택지를 하나의 choices(list)로 합쳐서 넣어주는 함수
    """
    choices = [data['A'], data['B'], data['C'], data['D']]
    return {'choices': choices}


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
    except Exception as e:
        if num_try > 100:
            with open("output.txt", "a") as f:
                f.write(data['question'] + "\n")
        time.sleep(3)
        add_summarized_question(data, num_try+1)
