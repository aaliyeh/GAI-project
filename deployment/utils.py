from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

def generate_text(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", 
                              max_length=128, truncation=True, padding="max_length").to(device)
            
    # Generate output
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=128,
    )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text