# KMMLU Summarization and Enhancement Project

## Project Overview
This project aims to enhance and analyze the [KMMLU dataset](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) through multiple approaches. The enhanced dataset is available at [KMMLU-Summarized-Chain_of_Thought](https://huggingface.co/datasets/SabaPivot/KMMLU-Summarized-Chain_of_Thought).

## Key Components

### Data Summarization

- Source: Original KMMLU dataset
- Method: Using Solar-Pro LLM
- Compression Rates:
  - Train: 10.13%
  - Dev: 8.51%
- Excluded Categories:
  - Electrical-Engineering
  - Electronics-Engineering
  - Industrial-Engineer

### Chain-of-Thought Generation

- Implementation: LangChain
- Target: Dev split only
- Purpose: Enable 5-shot chain-of-thought few-shot inference

### Model Evaluation

- Zero-shot learning
- 5-shot direct learning
- 5-shot CoT in-context learning

### QWEN Fine-tuning

- Model: QWEN 2.5-32B-it
- Method: Unsloth PEFT

### Solar-pro Fine-tuning

- Method: Huggingface LoRA

## Dataset Statistics
- Total size: 36.1 MB
- Number of rows: 207,251
- Monthly downloads: 542

## Results
| Category         | 0-shot | Direct Fewshot | CoT Fewshot | Average |
|------------------|--------|----------------|-------------|---------|
| Applied Science  | 51.0   | 55.7           | 55.8        | 54.2    |
| HUMSS            | 59.7   | 62.2           | 58.6        | 60.2    |
| Other            | 62.0   | 64.7           | 61.7        | 62.8    |
| STEM             | 54.4   | 61.7           | 59.1        | 58.4    |
| **Overall Average** | 56.1   | 61.2           | 58.7        | 58.7    |

To see the full result, please refer to the [results.md](results.md) file.

### Remarks
##### **Math (STEM Category)**
- **0-shot:** 32.0  
- **Direct Fewshot:** 65.0  
- **CoT Fewshot:** 82.0  
This domain has the **largest performance improvement** moving from 0-shot to CoT Fewshot, with a jump of **+50 points** from 32.0 to 82.0. This dramatic gain suggests that reasoning through CoT methods significantly benefits mathematical problem-solving.

#### **Korean-History (HUMSS Category)**
- **0-shot:** 37.0  
- **Direct Fewshot:** 34.0  
- **CoT Fewshot:** 29.0  
Korean-History shows a **performance decline** as it moves from 0-shot to CoT Fewshot. The drop from **37.0 to 29.0** (-8 points) suggests that the model struggles with complex reasoning or contextual continuity in this domain.



## License
This project inherits the license from the original KMMLU dataset.


## Citation
```bibtex
@misc{kmmlu_condensed,
  title = {Condensed KMMLU Dataset},
  author = {Saba Pivot},
  year = {2024},
  publisher = {Hugging Face},
  note = {Summarized and enhanced using Upstage's Solar-Pro LLM, including a chain of thought column.}
}
```

### Categories and Domains

#### 1. STEM
##### Science
- Agricultural-Sciences
- Biology
- Chemistry
- Ecology
- Environmental-Science
- Health
- Math

##### Technology
- Computer-Science
- Information-Technology
- Telecommunications-and-Wireless-Technology

#### Engineering
- Aviation-Engineering-and-Maintenance
- Chemical-Engineering
- Civil-Engineering
- Gas-Technology-and-Engineering
- Geomatics
- Maritime-Engineering
- Materials-Engineering
- Mechanical-Engineering
- Railway-and-Automotive-Engineering
- Refrigerating-Machinery

#### Mathematics
- Math
---

#### 2. Applied Science
- Food-Processing
- Fashion
- Construction
- Machine-Design-and-Manufacturing
- Energy-Management
- Nondestructive-Testing

---

#### 3. HUMSS (Humanities and Social Sciences)
- Accounting
- Criminal-Law
- Economics
- Education
- Korean-History
- Law
- Political-Science-and-Sociology
- Psychology
- Social-Welfare
- Marketing
- Real-Estate
- Taxation
- Public-Safety

---

#### 4. Other
- Interior-Architecture-and-Design
- Patent
- Management