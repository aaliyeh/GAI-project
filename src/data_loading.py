import pandas as pd
from datasets import Dataset

def load_and_clean_data(file_path: str) -> Dataset:
    """Load and preprocess CSV dataset."""
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")
    df = df.dropna(subset=["formal"])
    df["formal"] = df["formal"].str.strip()
    df = df[df["formal"] != ""]
    
    df.to_csv(file_path, index=False)
    return Dataset.from_pandas(df)

def get_dataset_splits(dataset: Dataset, test_size: float = 0.1):
    return dataset.train_test_split(test_size=test_size)
