from transformers import BertModel, BertTokenizer
from get_embeding import *

# Fixer les graines pour la reproductibilité
import torch
import numpy as np

#torch.manual_seed(42)
#np.random.seed(40)

# Charger un modèle pré-entraîné
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Mettre le modèle en mode évaluation
#model.eval()

# Tokenisation du texte
input_text = "Hello"
inputs = tokenizer(input_text, return_tensors="pt")

# Obtenir les embeddings
with torch.no_grad():  # Désactiver le calcul des gradients pour éviter des modifications accidentelles
    outputs = model(**inputs)
    embedding = outputs[0]

embedding = get_embedding_of_word(embedding)

print(embedding)

