import os
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from dotenv import load_dotenv
from utils import four_to_choice, load_model, add_summarized_question, add_chain_of_thought
from prompt import load_summarize_prompt, load_chain_of_thought_prompt

if __name__ == "__main__":
    load_dotenv()
    cpu_count = os.cpu_count()
    domains = ['Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology',
               'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction',
               'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering',
               'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing',
               'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer', 'Information-Technology',
               'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing', 'Management',
               'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering',
               'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety',
               'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare',
               'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math']
    domains = [d for d in domains if
               d not in ["Electrical-Engineering", "Electronics-Engineering", 'Industrial-Engineer']]
    llm_name = "solar-pro"
    model = load_model(llm_name)

    # To Summarize
    for category in domain:
        data = load_dataset("HAERAE-HUB/KMMLU", name=category)
        data_train = data['train'].map(four_to_choice)

        prompt = ChatPromptTemplate.from_messages(load_prompt(category))
        chain = prompt | model

        data_train = data_train.map(lambda x: add_summarized_question(x, chain=chain), num_proc=cpu_count)
        data_train.save_to_disk(f"summarized_{category}.hf")

    # To create CoT
    for category in domains:
        data = load_dataset("SabaPivot/KMMLU-Summarized-Chain_of_Thought", name=category)
        data_dev = data['dev'].map(four_to_choice)

        summarize_prompt = ChatPromptTemplate.from_messages(load_summarize_prompt(category))
        summarize_chain = summarize_prompt | model

        data_dev = data_dev.map(lambda x: add_summarized_question(x, chain=summarize_chain), num_proc=1)

        chain_of_thought_prompt = ChatPromptTemplate.from_messages(load_chain_of_thought_prompt(category))
        chain_of_thought_chain = chain_of_thought_prompt | model

        data_dev = data_dev.map(lambda x: add_chain_of_thought(x, chain=chain_of_thought_chain), num_proc=1)

        data_dev.save_to_disk(f"summarized_{category}.hf")

