from get_embeding import *
from transformers import BertModel, BertTokenizer, BertConfig, DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel
from tools import *
import matplotlib.pyplot as plt
import numpy as np


model_name = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
model = AutoModel.from_pretrained(model_name)

nb_error = nb_error(tokenizer, model, 100)
print("erreur : " + str(nb_error))