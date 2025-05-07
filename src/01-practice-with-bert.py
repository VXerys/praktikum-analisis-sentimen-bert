# Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model dan tokenizer
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased")

# Contoh teks
texts = [
    "Pelayanan di restoran ini sangat memuaskan, makanannya juga enak",
    "Kecewa dengan kualitas produk ini, tidak sesuai ekspektasi saya"
]

# Tokenize teks
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Prediksi
with torch.no_grad():
    outputs = model(**encoded_inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Hasil
for i, text in enumerate(texts):
    sentiment = "positif" if predictions[i, 1] > predictions[i, 0] else "negatif"
    print(f"Text: {text}")
    print(f"Sentimen: {sentiment} (Confidence: {max(predictions[i]).item():.4f})\n")