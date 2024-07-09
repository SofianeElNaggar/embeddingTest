import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from tools import *
import numpy as np

# Charger le tokenizer et le modèle BART pré-entraîné
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

vocab_size = len(tokenizer)
print(vocab_size)

# Exemple de texte d'entrée
input_text = "coffee"

# Tokeniser le texte d'entrée
inputs = tokenizer(input_text, return_tensors="pt")

encoder_outputs, embedding = get_embedding(inputs, model)

words = []
n = 1
m = 0

while len(words) <= 1:
    
    embeddings = rotate_around_point_lin(embedding, n*10)
    
    m = 0
    
    for e in embeddings:
        print(m)
        m += 1
        encoder_outputs.last_hidden_state[:, 1:2, :] = torch.FloatTensor(e)
        result = decode_embedding(encoder_outputs, model, tokenizer)
        words.append(result)
        print(result)

    words = list(set(words))
    n += 1
    print(words)
