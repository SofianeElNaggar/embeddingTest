from get_embeding import *
from transformers import BertModel, BertTokenizer, BertConfig, DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel
from tools import *
import matplotlib.pyplot as plt
import numpy as np

#tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
#model = BertModel.from_pretrained('bert-base-uncased')
#configuration = BertConfig(hidden_size=(12*32))
#model = BertModel(configuration) 

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
model = AutoModel.from_pretrained('microsoft/deberta-base')

model_name = "google-bert/bert-base-cased"


#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#model = DistilBertModel.from_pretrained("distilbert-base-uncased")

#tokenizer = BertTokenizer.from_pretrained(model_name)
#model = BertModel.from_pretrained(model_name)

nb_error = nb_error(tokenizer, model, 100)
print("erreur : " + str(nb_error))