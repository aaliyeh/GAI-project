# Russian Informal-to-Formal Text Translation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36.0-orange.svg)](https://huggingface.co/docs/transformers)

A sequence-to-sequence model for converting informal Russian text (e.g., "Приветик") into formal equivalents (e.g., "Здравствуйте").

## Project Overview
This project develops a Russian text formalization system using a fine-tuned T5 model to convert informal phrases (e.g., slang, chats) into formal equivalents. It addresses challenges like noisy outputs and lack of language understanding through a structured pipeline: dataset curation (35k+ pairs), model initialization with ai-forever/ruT5-base, and task-specific fine-tuning. While initial results show promising pattern recognition, the model struggles with rare slang and repetitive tokens, motivating plans for intermediate pretraining on raw Russian corpora. The repository includes modular code and a Jupyter notebook demonstrating end-to-end workflow, serving as a foundation for robust informal-to-formal translation in applications like email drafting or customer service

**Key Features**:
- Fine-tuned `ai-forever/ruT5-base` model
- Custom dataset of 35k+ informal-formal pairs
- ROUGE-L evaluation metrics
- Modular training pipeline

## Progress (10.03.25)
- **Current Status**:
  - Gathered a dataset of informal/formal sentences pairs
  - Initial fine-tuning complete 
    
- **Sample Output**:
  >Input: "Фига.....пытаешься извиниться, а им как будто все равно:("  
  >Output: "Пытаешься извиниться, а им как будто всё равно"

Our Russian-pretrained model still struggles to fully grasp the nuances of the language, often producing inconsistent or noisy outputs. While fine-tuning shows some improvement, the model’s limited understanding of proper Russian grammar and context continues to hold back its performance.
To address these limitations, as future plans we propose a two-phase language mastery framework:

#### Intermediate Pretraining Phase
- Objective: Strengthen foundational Russian comprehension
#### Refined Fine-Tuning Phase


This approach mirrors human language acquisition – first building broad linguistic competence, then specializing for formalization tasks.



## Project Structure
```
├── README.md
├── requirements.txt
├── src/
│ ├── data_loading.py # Dataset loading/cleaning
│ ├── model_training.py # Training logic
│ └── utils.py # Tokenization helpers
├── train.py # Main training script
└── notebooks/
└── GAI_project.ipynb # Jupyter notebook example
```


## Installation
```bash
git clone https://github.com/aaliyeh/GAI-project.git
cd GAI-project
pip install -r requirements.txt
```

## Usage
1. Training
``` bash
   python train.py \
  --model_name "ai-forever/ruT5-base" \
  --dataset_path "informal_formal.csv" \
  --epochs 3 \
  --batch_size 8
```

2. Inference
``` python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("./formalization_model")
tokenizer = T5Tokenizer.from_pretrained("./formalization_tokenizer")

input_text = "формализуй текст: Приветик! Как сам?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Jupyter Notebook Example
A complete interactive workflow is available in [`notebooks/GAI_project.ipynb`](notebooks/GAI_project.ipynb)

## Dataset
informal_formal.csv:
- 35,699 parallel sentences
- Sources: twitter, yandexGPT
- Columns: informal (original text), formal (target)
