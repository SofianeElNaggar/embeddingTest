from Bert.bert_huggingface.get_embedding import *
from transformers import BertModel, BertTokenizer, BertConfig, DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel
from tools import *
import matplotlib.pyplot as plt
import numpy as np


model_name = "google-bert/bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
model = AutoModel.from_pretrained(model_name)

#nb_error = nb_error(tokenizer, model, 100)
#print("erreur : " + str(nb_error))

vocab = tokenizer.get_vocab()

# Trier le vocabulaire par les indices
sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

# Enregistrer le vocabulaire dans un fichier texte
vocab_file_path = "bert_vocab.txt"
with open(vocab_file_path, "w") as f:
    for word, index in sorted_vocab:
        f.write(f"{word}\n")
