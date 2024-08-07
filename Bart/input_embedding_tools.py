from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import tools as tools
from collections import Counter
import numpy as np
from scipy.spatial.distance import euclidean
import pickle

def decode(model, tokenizer, input_ids, inputs_embeds):
    outputs = model.generate(input_ids, max_new_tokens=len(inputs_embeds[0]))
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def set_new_embedding(model, device, input_embeddings, input_ids, modified_embeds):
    vocab_size, embedding_dim = input_embeddings.weight.shape
    new_embeddings = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
    new_embeddings.weight.data.copy_(input_embeddings.weight.data)
    new_embeddings.weight.data[input_ids] = modified_embeds.squeeze(0).to(device)

    model.set_input_embeddings(new_embeddings)

def process_file(model, tokenizer, device, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    map = {}
    i = 0
    for line in lines:
        print(str(i) + "/50265")
        i += 1
        word = line.split(':')[0].strip()
        word = word.replace('Ġ', ' ')
        
        inputs = tokenizer(word, return_tensors='pt').to(device)

        input_ids = inputs['input_ids']
        input_embeddings = model.get_input_embeddings()
        inputs_embeds = input_embeddings(input_ids)
        embedding = inputs_embeds.cpu().detach().numpy()
        if len(embedding[0]) > 3:
            continue
        map[word] = embedding[0][1]
        
    return map

def get_vector(word, file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    return data.get(word, None)

def euclidean_distance(vec1, vec2):
    return euclidean(vec1, vec2)


def calculate_distances(target_word, file_path):
    # Charger le vecteur du mot cible
    target_vector = get_vector(target_word, file_path)
    
    if target_vector is None:
        raise ValueError(f"Le mot '{target_word}' n'a pas été trouvé dans le fichier.")
    
    # Charger tous les vecteurs depuis le fichier
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    distances = {}
    for word, vector in data.items():
        if word != target_word:
            distance = euclidean_distance(target_vector, vector)
            distances[word] = distance
    
    return distances

def get_closest_words(distances, n):
    # Trier le dictionnaire par distance croissante et obtenir les n premiers éléments
    closest_words = sorted(distances.items(), key=lambda item: item[1])[:n]
    
    # Extraire seulement les mots des n premiers éléments
    return [word for word, distance in closest_words]