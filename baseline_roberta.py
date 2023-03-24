import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load the RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Define the input text
input_text = "This drug is used to treat high blood pressure and heart failure."

# Tokenize the input text
input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0)

# Make a prediction
model.eval()
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs[0]
    predicted_label = torch.argmax(logits).item()

# Print the predicted label
if predicted_label == 0:
    print("The drug is used to treat high blood pressure.")
elif predicted_label == 1:
    print("The drug is used to treat heart failure.")
else:
    print("Unknown label.")
