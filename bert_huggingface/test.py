from transformers import EncoderDecoderModel, AutoTokenizer
import torch
from get_embeding import *

# Charger le modèle et le tokenizer
model_name = "patrickvonplaten/bert2bert_cnn_daily_mail"
model = EncoderDecoderModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Le mot ou la phrase dont vous voulez obtenir l'embedding
text = "example"

# Tokeniser le texte
inputs = tokenizer(text, return_tensors="pt")

# Obtenir les embeddings du texte avec l'encodeur
with torch.no_grad():
    encoder_outputs = model.encoder(**inputs)

# Récupérer le dernier état caché de l'encodeur
encoder_hidden_states = encoder_outputs[0]

print(encoder_outputs)


iterate_tokens(encoder_hidden_states, tokenizer, model)