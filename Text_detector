from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model from the saved directory
model_dir = "saved_model"  # Folder where the model is saved
tokenizer = AutoTokenizer.from_pretrained(model_dir)  # Load tokenizer from saved_model folder
model = AutoModelForSequenceClassification.from_pretrained(model_dir)  # Load the model

# Function to detect text (AI vs. Human)
def detect_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get the predicted label (0 for human, 1 for AI)
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    return "AI-generated" if predicted_class == 1 else "Human-generated"

# Example usage
if __name__ == "__main__":
    text = input("Enter text for detection: ")
    result = detect_text(text)
    print(f"Detected: {result}")
