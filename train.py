from src.data_loading import load_and_clean_data, get_dataset_splits
from src.model_training import initialize_model, get_trainer
from src.utils import preprocess_function
from datasets import load_dataset

def main():
    dataset = load_and_clean_data("informal_formal.csv")
    split_dataset = get_dataset_splits(dataset["train"])
    
    model, tokenizer = initialize_model()
    
    tokenized_train = split_dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )
    tokenized_val = split_dataset["test"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )
    
    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        epochs=1,
        batch_size=8
    )
    trainer.train()
    
    model.save_pretrained("./formalization_model")
    tokenizer.save_pretrained("./formalization_tokenizer")

if __name__ == "__main__":
    main()
