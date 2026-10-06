
import re
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# Text Preprocessing
# --------------------------------------------------

def preprocess_text1(text):

    # Handle missing values
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 4. Remove escaped characters
    text = text.replace("\\'", "'")
    text = text.replace('\\"', '"')

    # 5. Remove punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 6. Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # 7. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --------------------------------------------------
# Hugging Face Model
# --------------------------------------------------

MODEL_PATH = "rajib-ranjan-9861/distilbert-ai-detector"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()

import re
import pandas as pd

def preprocess_text(text):

    # Handle missing values
    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Keep only alphabets, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_text(text):

    # Preprocess input text
    text = preprocess_text1(text)
    text=preprocess_text(text)

    # Tokenization
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_token_type_ids=False
    )

    # Move tensors to device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Model prediction
    with torch.no_grad():

        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    ai_prob = float(probabilities[0])
    human_prob = float(probabilities[1])

    prediction = 0 if ai_prob >= human_prob else 1

    label = "AI" if prediction == 0 else "Human"

    return {
        "prediction": prediction,
        "label": label,
        "ai_probability": ai_prob,
        "human_probability": human_prob
    }


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🤖 AI Text Detector")

st.write(
    "Enter a piece of text to determine whether it is "
    "more likely to be AI-generated or human-written."
)


text = st.text_area(
    "Enter your text",
    height=250,
    placeholder="Write or paste your text here..."
)


# --------------------------------------------------
# Detect Button
# --------------------------------------------------

if st.button("🔍 Detect", use_container_width=True):

    if not text.strip():

        st.warning("Please enter some text.")

    else:

        result = predict_text(text)

        st.divider()

        st.subheader("Prediction")

        if result["label"] == "AI":
            st.error("🤖 AI Generated")
        else:
            st.success("👤 Human Written")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "AI Probability",
                f"{result['ai_probability'] * 100:.5f}%"
            )

        with col2:

            st.metric(
                "Human Probability",
                f"{result['human_probability'] * 100:.5f}%"
            )

        st.progress(
            result["ai_probability"]
        )