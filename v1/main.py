import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from tools import *

# Charger le tokenizer et le modèle BART pré-entraîné
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Exemple de texte d'entrée
input_text = "Hello"

# Tokeniser le texte d'entrée
inputs = tokenizer(input_text, return_tensors="pt")

encoder_outputs, embedding = get_embedding(inputs, model)

print(embedding)

embedding = modif_embedding(embedding)

print(embedding)

encoder_outputs.last_hidden_state[:, 1:2, :] = torch.FloatTensor(embedding)

result = decode_embedding(encoder_outputs, model, tokenizer)

print(result)
