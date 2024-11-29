import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the saved model and tokenizer
model_dir = "saved_model"  # Directory containing the model
tokenizer = AutoTokenizer.from_pretrained(model_dir)  # Load tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_dir)  # Load model

# Function to detect text (AI vs. Human)
def detect_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    return "AI-generated" if predicted_class == 1 else "Human-generated"

# Streamlit app layout
st.set_page_config(page_title="Text Detection", page_icon="📝", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5rem;
        color: #4B0082;
        margin-top: 20px;
    }
    .description {
        text-align: center;
        font-size: 1.2rem;
        color: #555555;
    }
    .input-area {
        margin: 20px auto;
        padding: 20px;
        border: 2px solid #4B0082;
        border-radius: 10px;
        background-color: #F9F9F9;
    }
    .result {
        font-size: 1.5rem;
        color: #228B22;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="title">Text Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Enter text to classify as AI-generated or Human-generated:</p>', unsafe_allow_html=True)

# Input text box with a stylish container
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    input_text = st.text_area("Input Text", height=150, key="input_text", placeholder="Type your text here...")
    st.markdown('</div>', unsafe_allow_html=True)

# Button to classify the text
if st.button("Detect", key="detect_button"):
    if input_text:
        result = detect_text(input_text)
        st.markdown(f'<p class="result">Detected: <strong>{result}</strong></p>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text.")
