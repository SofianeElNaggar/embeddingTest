import torch
from scipy.spatial.distance import cosine

# Fonction pour obtenir l'embedding d'une phrase
def get_bert_embedding(text, tokenizer, model):
    # Tokenisation du texte
    inputs = tokenizer(text, return_tensors="pt")
    # Obtenir les embeddings
    with torch.no_grad(): 
        outputs = model(**inputs)
        embedding = outputs[0]
    return embedding

# Supprime les classifier token de début et de fin
# A utiliser pour les embedding d'un mot unique
def get_embedding_of_word(embedding):
    return embedding[:, 1:2, :]

# Fonction pour retrouver le mot à partir de l'embedding
# Ne pas appeler - Utiliser iterate_tokens()
def retrieve_word_from_embedding(embedding, tokenizer, model):
    # Convertir l'embedding en numpy array pour la mesure de similarité
    embedding = embedding.cpu().detach().numpy()[0]
    
    # Récupérer le vocabulaire du tokenizer
    vocab = tokenizer.get_vocab()

    min_distance = float('inf')
    closest_token = None

    for token, index in vocab.items():
        token_embedding = model.embeddings.word_embeddings.weight[index].detach().cpu().numpy()

        # Calculer la distance cosinus entre l'embedding donné et le token_embedding
        distance = cosine(embedding, token_embedding)

        if distance < min_distance:
            min_distance = distance
            closest_token = token
    
    # Utiliser le tokenizer pour récupérer le mot correspondant au token trouvé
    word = tokenizer.convert_tokens_to_string([closest_token])

    return word 

def iterate_tokens(embedding, tokenizer, model):
    words = []
    for x in range(embedding.shape[1]):
        e = embedding[:, x, :]
        words.append(retrieve_word_from_embedding(e, tokenizer, model))
    return words