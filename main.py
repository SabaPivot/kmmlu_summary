import os
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
from dotenv import load_dotenv
from utils import four_to_choice, load_model, add_summarized_question
from prompt import load_prompt

if __name__ == "__main__":
    load_dotenv()
    cpu_count = os.cpu_count()

    domain = ['Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering', 'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing', 'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer', 'Information-Technology', 'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing', 'Management', 'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering', 'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety', 'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare', 'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math']

    llm_name = "solar-pro"
    model = load_model(llm_name)

    for category in domain:
        data = load_dataset("HAERAE-HUB/KMMLU", name=category)
        data_train = data['train'].map(four_to_choice)

        prompt = ChatPromptTemplate.from_messages(load_prompt(category))
        chain = prompt | model

        data_train = data_train.map(lambda x: add_summarized_question(x, chain=chain), num_proc=cpu_count)
        data_train.save_to_disk(f"summarized_{category}.hf")
