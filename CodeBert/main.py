from transformers import RobertaTokenizer, RobertaModel
import torch
from get_embedding import *

# Charger le tokenizer et le modèle CodeBERT
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

# Exemple de texte d'entrée
input_text = "hello"

embedding = get_bert_embedding(input_text, tokenizer, model)

output = iterate_tokens(embedding, tokenizer, model)

print(output)
