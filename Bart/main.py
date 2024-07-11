from transformers import BartForConditionalGeneration, BartTokenizer
from tools import *
import numpy as np


model = "facebook/bart-large"

tokenizer = BartTokenizer.from_pretrained(model)
model = BartForConditionalGeneration.from_pretrained(model)

random_interpolate_test(tokenizer, model, 10)

#print(interpolate_test(tokenizer, model, "king", "king"))

#find_neighbor_around(embedding, encoder_outputs, model, tokenizer, step=1, start_distance=1, min_lap=3)

