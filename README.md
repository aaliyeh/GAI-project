# Russian Informal-to-Formal Text Translation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36.0-orange.svg)](https://huggingface.co/docs/transformers)

A sequence-to-sequence model for converting informal Russian text (e.g., "Приветик") into formal equivalents (e.g., "Здравствуйте").

## Project Overview
This project develops a sequence-to-sequence model based on T5 (Text-to-Text Transfer Transformer) to automatically convert informal Russian phrases (colloquialisms, slang, chat language) into grammatically correct, formal equivalents. The system addresses key NLP challenges:

- Noise robustness (handling typos, abbreviations)

- Contextual understanding (preserving meaning while formalizing)

- Low-resource adaptation (effective training on limited parallel data)

 **Key Features88
- Fine-tuned T5-base (~220M params) optimized for Russian text formalization
- Curated dataset of 35,000+ informal-formal phrase pairs

## Project Structure
```
├── README.md
├── deployment/
│ ├── app.py                    # streamlit app
│ ├── utils.py                  # helper functions
│ └── requirements.txt
├── notebooks/                  #data analysis and experiments
| ├── model_pretraining.ipynb
| ├── model_fine_tuning.ipynb
| └── GAI_EDA.ipynb
├── models/                     # models files
| ├── fine_tuned_model/
| | ├──  . . .
| ├── rut5_model/
| | ├──  . . .
└── Dockerfile
```


## Installation
```bash
git clone https://github.com/aaliyeh/GAI-project.git
cd GAI-project
```
## Downloading the models
To try the service locally on your machine, you will need to download the models:
1. Our [fine-tuned model](https://drive.google.com/file/d/1l1u3knhtYZZLsSXNR2XmguT4bsxeSC0O/view?usp=drive_link), the result of this project
2. Basic [T5 model](https://drive.google.com/file/d/1cDaZs1oP5etsCDn5sYjmq3eCXConcJO8/view?usp=drive_link) for comparison reasons

Unzipped folders `fine_tuned_model` and `rut5_model` are need to be located in `models` directory:
```bash
mkdir models
mv fine_tuned_model models/
mv rut5_model models/
```

## Usage
You can either run streamlit directly on your computer or user docker container
-  To run stramlit directly:
```bash
streamlit run deployment/app.py
```
- to run docker container:
```bash
docker build -t informal2formal .
docker run -p 8502:8502 informal2formal
```

## Jupyter Notebook Example
A complete interactive workflow is available in [`notebooks/GAI_project.ipynb`](notebooks/GAI_project.ipynb)

## Dataset
informal_formal.csv:
- 35,699 parallel sentences
- Sources: twitter, chatGPT, yandexGPT
- Columns: informal (original text), formal (target)

## 🛠 Future Work
- Expand dataset to 100k+ pairs (focusing on niche slang)

- Experiment with T5-large (770M params)

- Add informality scoring (0-1 scale quantifying casualness)

- Deploy as Telegram/WhatsApp bot

