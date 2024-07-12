from Bert.bert_huggingface.get_embedding import *
from transformers import BertModel, BertTokenizer, BertConfig
from tools import *
import matplotlib.pyplot as plt
import numpy as np

input_text = "hello"

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")

#test_hidden_size(input_text, tokenizer,10,5,64)

#print("input=hello - h_size=12*32")
#test_distance(tokenizer, input_text, n=2000,h_size=(12*32))
#print("input=random - h_size=12*32")
#test_distance(tokenizer, n=2000, random=True, h_size=(12*32))
#print("input=hello - h_size=12*64")
#test_distance(tokenizer, input_text, n=2000,h_size=(12*64))
#print("input=random - h_size=12*64")
#test_distance(tokenizer, n=2000, random=True, h_size=(12*64))

configuration = BertConfig()
model = BertModel(configuration) 
#model = BertModel.from_pretrained('bert-base-uncased')

print(model)
