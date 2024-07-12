from transformers import BartForConditionalGeneration, BartTokenizer
from tools import *
import numpy as np


model = "facebook/bart-large"

tokenizer = BartTokenizer.from_pretrained(model)
model = BartForConditionalGeneration.from_pretrained(model)

#random_interpolate_test(tokenizer, model, 150)



input_text = "hello"
inputs = tokenizer(input_text, return_tensors="pt")
encoder_outputs, embedding = get_embedding(inputs, model)

print(encoder_outputs.last_hidden_state)

#print(inputs)

"""
input_text2 = ""
inputs2 = tokenizer(input_text2, return_tensors="pt")
encoder_outputs2, embedding2 = get_embedding(inputs2, model)

t = torch.cat((encoder_outputs.last_hidden_state[0][:1], encoder_outputs.last_hidden_state[0][2:]))

#print(encoder_outputs2)

encoder_outputs2.last_hidden_state[0] = t

#print(encoder_outputs2)
"""

#print(decode_embedding(encoder_outputs, model, tokenizer))


#find_neighbor_around(embedding, encoder_outputs, model, tokenizer, step=1, start_distance=1, min_lap=3)

