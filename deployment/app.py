import streamlit as st
import torch
import asyncio
import nest_asyncio
from utils import load_model, generate_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not hasattr(asyncio, '_get_running_loop'):
    asyncio._get_running_loop = asyncio.get_running_loop

# Set page config
st.set_page_config(page_title="intormal2formal", page_icon="🤖")

model_options = {
    "Base Model": "models/rut5_model",
    "Fine-tuned Model": "models/fine_tuned_model"
}
selected_model = st.selectbox("Choose model:", list(model_options.keys()))


tokenizer_base, model_base = load_model(model_options['Base Model'])
tokenizer_custom, model_custom = load_model(model_options['Fine-tuned Model'])

model_tok = {'Base Model': (tokenizer_base, model_base), 'Fine-tuned Model': (tokenizer_custom, model_custom)}

# App title and description
st.title("Informal to formal converter")
st.write("This application converts informal russian text to more formal version")

# Input section
input_text = st.text_area("Enter your input text:", height=150, 
                         placeholder="Type or paste your text here...")

# Generate button
if st.button("Generate"):
    if input_text.strip() == "":
        st.warning("Please enter some input text.")
    else:
        with st.spinner("Generating..."):

            tokenizer, model = model_tok[selected_model]
            generated_text = generate_text(model, tokenizer, input_text)
        
            st.subheader("Generated Text:")
            st.write(generated_text)
