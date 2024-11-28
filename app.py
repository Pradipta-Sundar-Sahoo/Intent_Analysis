import streamlit as st
import torch
import pandas as pd
import openai
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    print("Error: OPENAI_API_KEY is not set in the .env file.")
else:
    openai.api_key = OPENAI_API_KEY
    print("OpenAI API Key loaded successfully!")

from streamlit_lottie import st_lottie
import json

def load_lottie_url(url: str):
    with open(url, "r") as f:
        return json.load(f)
lottie_file = 'Animation - 1732745092104.json'

col1, col2= st.columns([1, 1])

with col2:
    st_lottie(load_lottie_url(lottie_file), speed=2, width=200, height=200, key="animation")

with col1:
    st.markdown("<h1 style='text-align: left; color: #4CAF50;'>Text Classification Model Selector</h1>", unsafe_allow_html=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = 'train.csv'
data = pd.read_csv(file_path)
label_to_index = {label: i for i, label in enumerate(data['label'].unique())}
index_to_label = {i: label for label, i in label_to_index.items()}
candidate_labels = list(label_to_index.keys())

st.markdown("""
    <style>
        .stTextInput textarea {
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #4CAF50;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px 24px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stSelectbox select {
            font-size: 16px;
            padding: 8px;
            border-radius: 8px;
            border: 1px solid #4CAF50;
        }
        .stWrite {
            font-size: 18px;
            color: #333;
            padding: 10px;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)

model_choice = st.selectbox("Select Model", ["DistilBERT", "RoBERTa", "BERT", "OpenAI"])

user_input = st.text_area("Enter your query:")

submit_button = st.button("Submit")

def load_model(model_choice: str):
    if model_choice == "RoBERTa":
        model_path = './roberta_intent_model'
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    elif model_choice == "DistilBERT":
        model_path = './distilbert_intent_model'
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    elif model_choice == "BERT":
        model_path = './bert_intent_model'
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    else:
        tokenizer = None
        model = None
    
    model.to(device)
    model.eval()
    return tokenizer, model

def predict_intent_with_transformers(sentence, tokenizer, model):
    encoding = tokenizer(
        sentence, max_length=128, padding='max_length', truncation=True, return_tensors="pt"
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return index_to_label[predicted_class]

def predict_intent_with_openai(sentence, labels):
    openai.api_key = "<your_openai_api_key>"

    prompt = f"The following text: \"{sentence}\" belongs to one of these categories: {', '.join(labels)}. Identify the most suitable category."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for text classification. Give the output of the correct label only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        predicted_label = response['choices'][0]['message']['content'].strip()
        return predicted_label
    except Exception as e:
        st.error(f"Error: {e}")
        return None

if submit_button and user_input:
    if model_choice != "OpenAI":
        tokenizer, model = load_model(model_choice)
        predicted_intent = predict_intent_with_transformers(user_input, tokenizer, model)
        st.markdown(f"<div class='stWrite'><strong>Predicted Intent:</strong> {predicted_intent}</div>", unsafe_allow_html=True)
    else:
        predicted_label = predict_intent_with_openai(user_input, candidate_labels)
        st.markdown(f"<div class='stWrite'><strong>Predicted Label:</strong> {predicted_label}</div>", unsafe_allow_html=True)
