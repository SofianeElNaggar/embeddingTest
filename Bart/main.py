from transformers import BartForConditionalGeneration, BartTokenizer
from tools import *

# Charger le tokenizer et le modèle BART pré-entraîné
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

input_text = "hello"

inputs = tokenizer(input_text, return_tensors="pt")

encoder_outputs, embedding = get_embedding(inputs, model)


"""
input_text2 = "hello"

inputs2 = tokenizer(input_text2, return_tensors="pt")

encoder_outputs2, embedding2 = get_embedding(inputs2, model)

print(interpolate_vectors(embedding, embedding2, 10))
"""

print(embedding)

print(decode_embedding(encoder_outputs, model, tokenizer))

find_neighbor_around(embedding, encoder_outputs, model, tokenizer, neighbor_number=1, step=1, start_distance=1, min_lap=3)

