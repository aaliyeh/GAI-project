from transformers import T5Tokenizer

def preprocess_function(
    examples,
    tokenizer: T5Tokenizer,
    task_prefix: str = "формализуй текст: ",
    max_length: int = 128
):
    inputs = [task_prefix + ex for ex in examples["informal"]]
    targets = [ex for ex in examples["formal"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
