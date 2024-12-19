from datasets import load_dataset, load_from_disk, DatasetDict

domains = ['Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering', 'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing', 'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer', 'Information-Technology', 'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing', 'Management', 'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering', 'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety', 'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare', 'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math']
domains = [d for d in domains if d not in ["Electrical-Engineering", "Electronics-Engineering", 'Industrial-Engineer']]
for domain in domains:
    train_dataset = load_from_disk(f"/home/careforme.dropout/kmmlu_summary/summarized_{domain}.hf")
    train_dataset = train_dataset.remove_columns(["choices", "question"])
    train_dataset = train_dataset.rename_column("summarized_q", "question")

    dev_dataset = load_dataset("HAERAE-HUB/KMMLU", name=domain)["dev"]
    test_dataset = load_dataset("HAERAE-HUB/KMMLU", name=domain)["test"]

    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset
        }
    )
    print(train_dataset)
    print(dataset_dict)
    dataset_dict.push_to_hub("SabaPivot/KMMLU-Summarized-Chain_of_Thought", config_name=domain)
