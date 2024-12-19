import subprocess
import yaml
import os

yaml_file = "trainer.yaml"
log_file = "inference_logs.txt"

domains = [
    # "Accounting",
    # "Agricultural-Sciences",
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
    with open(log_file, "a") as log:
        log.write(f"\n--- Updated YAML for domain: {domain} ---\n")

def execute_commands():
    with open(log_file, "w") as log:
        log.write("Inference Logs\n" + "=" * 50 + "\n")
    
    for domain in domains:
        update_yaml_domain(domain)
        with open(log_file, "a") as log:
            log.write(f"\nResults for domain: **{domain}**\n{'-' * 40}\n")
        
        for description, command in commands:
            with open(log_file, "a") as log:
                log.write(f"Running {description}...\n")
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                output = f"{description}: ✅ Success\nOutput:\n{result.stdout.strip()}"
            else:
                output = f"{description}: ❌ Failed\nError:\n{result.stderr.strip()}"

            with open(log_file, "a") as log:
                log.write(output + "\n" + "-" * 40 + "\n")
                print(output)

if __name__ == "__main__":
    execute_commands()