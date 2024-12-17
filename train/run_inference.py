import subprocess
import yaml
import os

yaml_file = "trainer.yaml"

domains = [
    "Accounting",
    "Agricultural-Sciences",
    "Aviation-Engineering-and-Maintenance",
    "Biology",
    "Chemical-Engineering",
    "Chemistry",
    "Civil-Engineering",
    "Computer-Science",
    "Construction",
    "Criminal-Law",
    "Ecology",
    "Economics",
    "Education",
    "Energy-Management",
    "Environmental-Science",
    "Fashion",
    "Food-Processing",
    "Gas-Technology-and-Engineering",
    "Geomatics",
    "Health",
    "Information-Technology",
    "Interior-Architecture-and-Design",
    "Korean-History",
    "Law",
    "Machine-Design-and-Manufacturing",
    "Management",
    "Maritime-Engineering",
    "Marketing",
    "Materials-Engineering",
    "Math",
    "Mechanical-Engineering",
    "Nondestructive-Testing",
    "Patent",
    "Political-Science-and-Sociology",
    "Psychology",
    "Public-Safety",
    "Railway-and-Automotive-Engineering",
    "Real-Estate",
    "Refrigerating-Machinery",
    "Social-Welfare",
    "Taxation",
    "Telecommunications-and-Wireless-Technology"
]

commands = [
    ("0-shot", ["python", "main.py", "--mode", "inference"]),
    ("Direct fewshot", ["python", "main.py", "--mode", "inference", "--fewshot"]),
    ("CoT fewshot", ["python", "main.py", "--mode", "inference", "--cot"]),
]

def update_yaml_domain(domain):
    """Update the 'domain' field in the YAML file."""
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    config['dataset']['domain'] = domain
    with open(yaml_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"\n--- Updated YAML for domain: {domain} ---")

def execute_commands():
    for domain in domains:
        update_yaml_domain(domain)
        print(f"\nResults for domain: **{domain}**\n{'-' * 40}")
        
        for description, command in commands:
            print(f"Running {description}...")
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"{description}: ✅ Success\nOutput:\n{result.stdout.strip()}")
            else:
                print(f"{description}: ❌ Failed\nError:\n{result.stderr.strip()}")
            print("-" * 40)

if __name__ == "__main__":
    execute_commands()
