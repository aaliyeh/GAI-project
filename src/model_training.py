from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np

def initialize_model(model_name: str = "ai-forever/ruT5-base"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def compute_rouge(tokenizer, rouge):
    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        return rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            rouge_types=["rougeL"]
        )
    return _compute_metrics

def get_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir="./results",
    learning_rate=2e-5,
    batch_size=8,
    epochs=1
):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        predict_with_generate=True,
    )

    rouge = evaluate.load("rouge")
    
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_rouge(tokenizer, rouge),
    )
