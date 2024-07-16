import torch
import random
import itertools
import math
from collections import Counter
import json
import numpy as np

def get_embedding(inputs, model):
    with torch.no_grad():
        encoder_outputs = model.model.encoder(**inputs)
        embedding = encoder_outputs.last_hidden_state
    return encoder_outputs, embedding[0][1:-1].cpu().detach().numpy()

def batch_decode_embedding(encoder_outputs, model, tokenizer):
    # Initialiser l'entrée du décodeur avec le token de début de séquence
    decoder_start_token = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token]], device=encoder_outputs.last_hidden_state.device)
    
    # Générer la séquence de sortie en utilisant les embeddings de l'encodeur
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            max_length=50,
            num_beams=1,
            early_stopping=False
        )
    
    # Décoder les tokens de sortie en texte lisible
    result = tokenizer.batch_decode(output_sequences[0], skip_special_tokens=True)
    
    return result

def decode_embedding(encoder_outputs, model, tokenizer):
    # Initialiser l'entrée du décodeur avec le token de début de séquence
    decoder_start_token = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token]], device=encoder_outputs.last_hidden_state.device)
    
    # Générer la séquence de sortie en utilisant les embeddings de l'encodeur
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            max_length=50,
            num_beams=1,
            early_stopping=False
        )
    
    # Décoder les tokens de sortie en texte lisible
    result = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return result

#Sélection un mot aléatoire dans un fichier
def random_word():
    fichier = './english-common-words.txt'
    with open(fichier, 'r') as f:
        lignes = f.readlines()
        ligne = random.choice(lignes)
    return ligne.strip()

def generate_positions(n, distance):
    # Generate all possible combinations of -1, 0, and 1 for n dimensions
    directions = list(itertools.product([-1, 0, 1], repeat=n))
    # Remove the origin (0, 0, ..., 0) since it's not a valid direction
    directions = [d for d in directions if any(d)]
    
    # Normalize directions to have the correct distance
    positions = []
    for direction in directions:
        length = math.sqrt(sum(d**2 for d in direction))
        normalized_direction = [d * distance / length for d in direction]
        positions.append(normalized_direction)
    
    return positions

def rotate_around_point_lin(vector, distance):
    n = len(vector)
    rotated_vectors = []

    # Positions relatives autour du point central
    positions = []
    # Directions cardinales
    for i in range(n):
        pos = [0] * n
        pos[i] = distance
        positions.append(pos)
        pos = [0] * n
        pos[i] = -distance
        positions.append(pos)

    # Diagonales (en 2D)
    if n == 2:
        diagonal_distance = distance / math.sqrt(2)
        positions.extend([
            [diagonal_distance, diagonal_distance],
            [-diagonal_distance, diagonal_distance],
            [diagonal_distance, -diagonal_distance],
            [-diagonal_distance, -diagonal_distance]
        ])

    # Calculer les vecteurs tournés
    for pos in positions:
        rotated_vector = [vector[i] + pos[i] for i in range(n)]
        rotated_vectors.append(rotated_vector)

    return rotated_vectors

#Regarde autour d'un mot 
#neighbor_number : le nombre de voisin minimum à trouver pour stoper le programme
#step : distance d'incrémentation du cercle à chaque itération
#start_distance : distance de départ de l'incrémentation du cecle (si 0, l'incrémentation commencera à une distance de step)
#min_lap : le nombre d'incrémentation minimum du cercle autour du point d'origine
#max_lap : le nombre d'incrémentation maximum du cercle autour du point d'origine
def find_neighbor_around(embedding, encoder_outputs, model, tokenizer, device, neighbor_number=1, step=0.5, start_distance=0, min_lap=0, max_lap=1):
    n = 0
    
    if start_distance == 0:
        start_distance = step
    
    if min_lap > max_lap:
        max_lap = min_lap
        
    words = {}
    distance = start_distance
    
    while (len(words) <= neighbor_number and n <= min_lap) or max_lap > n:
        
        list_embeddings = []
        for i in range(len(embedding)):
            embeddings = rotate_around_point_lin(embedding[i], distance)
            list_embeddings.append(embeddings)
            
        results = []
        for i in range(len(list_embeddings[0])):
            for j in range(len(list_embeddings)):
                encoder_outputs.last_hidden_state[:, 1+j:2+j, :] = torch.FloatTensor(list_embeddings[j][i]).to(device)
                result = decode_embedding(encoder_outputs, model, tokenizer)
                results.append(result)
            
        words[distance] = dict(Counter(results))
        distance += step
        n += 1

    return words

def interpolate_vectors(v1, v2, n):
    vectors = []
    for i in range(n):
        alpha = i / (n - 1)  # interpolation parameter from 0 to 1
        interpolated_vector = (1 - alpha) * v1 + alpha * v2
        vectors.append(interpolated_vector)
    return vectors

def interpolate_test(tokenizer, model, device, input_text1, input_text2, n=100):
    input_text = "a"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    encoder_outputs, embedding = get_embedding(inputs, model)

    inputs1 = tokenizer(input_text1, return_tensors="pt").to(device)
    encoder_outputs1, embedding1 = get_embedding(inputs1, model)

    inputs2 = tokenizer(input_text2, return_tensors="pt").to(device)
    encoder_outputs2, embedding2 = get_embedding(inputs2, model)
    
    assert len(embedding1) == 1 and len(embedding2) == 1

    interpolated_vectors = interpolate_vectors(embedding1, embedding2, n)

    result = []
    result.append(input_text1)
    for e in interpolated_vectors:
        encoder_outputs.last_hidden_state[:, 1:2, :] = torch.FloatTensor(e).to(device)
        result.append(decode_embedding(encoder_outputs, model, tokenizer))
        
    result.append(input_text2)
    result = dict(Counter(result))
    return result

def random_interpolate_test(tokenizer, model, device, nb_result, nb_point=100):
    result = []
    while len(result) < nb_result:
        
        input_text = "a"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        encoder_outputs, embedding = get_embedding(inputs, model)
        
        input_text1 = random_word()
        inputs1 = tokenizer(input_text1, return_tensors="pt").to(device)
        encoder_outputs1, embedding1 = get_embedding(inputs1, model)
        
        while len(embedding1) != 1:
            input_text1 = random_word()
            inputs1 = tokenizer(input_text1, return_tensors="pt").to(device)
            encoder_outputs1, embedding1 = get_embedding(inputs1, model)
            
        input_text2 = random_word()
        inputs2 = tokenizer(input_text2, return_tensors="pt").to(device)
        encoder_outputs1, embedding2 = get_embedding(inputs2, model)
        
        while len(embedding2) != 1:
            input_text2 = random_word()
            inputs2 = tokenizer(input_text2, return_tensors="pt").to(device)
            encoder_outputs1, embedding2 = get_embedding(inputs2, model)
            
        interpolated_vectors = interpolate_vectors(embedding1, embedding2, nb_point)
        
        r = []
        r.append(input_text1)
        for e in interpolated_vectors:
            encoder_outputs.last_hidden_state[:, 1:2, :] = torch.FloatTensor(e).to(device)
            r.append(decode_embedding(encoder_outputs, model, tokenizer))
        r.append(input_text2)
        
        r = dict(Counter(r))
        if len(r)>=3:
            result.append(r)
            print(str(len(result)) + "/" + str(nb_result))
    
           
    with open('result.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)    
        
def distance_euclid_between_vectors(liste_vecteurs1, liste_vecteurs2):
    distances = []
    for v1, v2 in zip(liste_vecteurs1, liste_vecteurs2):
        # Convertir les listes en tableaux numpy pour faciliter les calculs
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        # Calculer la distance euclidienne entre les deux vecteurs
        distance = np.linalg.norm(v1_np - v2_np)
        distances.append(distance)
    return distances

def cosinus_distance(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_sim = dot_product / (norm_v1 * norm_v2)
    cos_dist = 1 - cos_sim
    return cos_dist

def distance_cosinus_between_vectors(liste_vecteurs1, liste_vecteurs2):
    if len(liste_vecteurs1) != len(liste_vecteurs2):
        raise ValueError("Les deux listes de vecteurs doivent être de même longueur.")
    
    distances = []
    for v1, v2 in zip(liste_vecteurs1, liste_vecteurs2):
        distance = cosinus_distance(v1, v2)
        distances.append(distance)
    
    return distances

def check_tokenization_integrity(sentence, tokenized_sentence):
    original_words = sentence.split()
    tokenized_sentence = tokenized_sentence
    
    # Enlever les tokens spéciaux si présents
    if tokenized_sentence[0] == '</s>' and tokenized_sentence[1] == '<s>':
        tokenized_sentence = tokenized_sentence[2:-1]
    
    reconstructed_words = []
    current_word = ""
    
    for token in tokenized_sentence:
        if token.startswith("▁") or token.startswith("##") or token.startswith(" "):
            if current_word:
                reconstructed_words.append(current_word.strip())
            current_word = token
        else:
            current_word += token
    
    # Ajouter le dernier mot reconstruit
    if current_word:
        reconstructed_words.append(current_word.strip())
    
    if len(original_words) != len(reconstructed_words):
        return False
    
    for orig_word, recon_word in zip(original_words, reconstructed_words):
        if orig_word != recon_word:
            return False
    
    return True

def find_different_word_index(sentence1, sentence2):
    # Split each sentence into lists of words
    words1 = sentence1.split()
    words2 = sentence2.split()
    
    # Compare word by word
    for i in range(len(words1)):
        if words1[i] != words2[i]:
            return i  # Return the index of the first different word
    
    # If no difference found (should not happen if assumptions are met)
    return -1

def calculate_distances_and_indices(json_file, device, tokenizer, model):
    results = []
    
    # Charger le fichier JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for pair in data:
        sentence1 = pair['sentence1']
        sentence2 = pair['sentence2']
        
        # Obtenir les embeddings pour sentence1
        inputs1 = tokenizer(sentence1, return_tensors="pt").to(device)
        encoder_outputs1, embedding1 = get_embedding(inputs1, model)
        e1 = encoder_outputs1.last_hidden_state[0].cpu().detach().numpy()
        
        # Obtenir les embeddings
        inputs2 = tokenizer(sentence2, return_tensors="pt").to(device)
        encoder_outputs2, embedding2 = get_embedding(inputs2, model)
        e2 = encoder_outputs2.last_hidden_state[0].cpu().detach().numpy()
        
        b1 = check_tokenization_integrity(sentence1, batch_decode_embedding(encoder_outputs1, model, tokenizer))
        b2 = check_tokenization_integrity(sentence2, batch_decode_embedding(encoder_outputs2, model, tokenizer))
        
        if not b1 or not b2:
            continue
        
        if len(e1) != len(e2):
            continue
        
        # Calculer la distance entre les vecteurs d'embedding
        distances_euclid = distance_euclid_between_vectors(e1, e2)
        distances_cosinus = distance_cosinus_between_vectors(e1, e2)
        
        # Trouver l'indice du premier mot différent
        idx_different_word = find_different_word_index(sentence1, sentence2)
        
        # Créer un résultat pour cette paire
        result = {
            "distance_euclid": distances_euclid,
            "distance_cosinus": distances_cosinus,
            "different_word_index": idx_different_word+1
        }
        
        # Ajouter le résultat à la liste des résultats
        results.append(result)
    
    return results

def distance_between_random_words(device, model, tokenizer, n):
    
    results = []
    
    for i in range(n):
        word1 = random_word()
        inputs = tokenizer(word1, return_tensors="pt").to(device)
        encoder_outputs, embedding = get_embedding(inputs, model)
        e1 = encoder_outputs.last_hidden_state[0].cpu().detach().numpy()
        
        word2 = random_word()
        inputs = tokenizer(word2, return_tensors="pt").to(device)
        encoder_outputs2, embedding2 = get_embedding(inputs, model)
        e2 = encoder_outputs2.last_hidden_state[0].cpu().detach().numpy()
        
        if len(e1) != 3 or len(e2) !=3:
            continue
        
        # Calculer la distance entre les vecteurs d'embedding
        distances_euclid = distance_euclid_between_vectors(e1, e2)
        distances_cosinus = distance_cosinus_between_vectors(e1, e2)
        
        # Créer un résultat pour cette paire
        result = {
            "distance_euclid": distances_euclid,
            "distance_cosinus": distances_cosinus,
            "different_word_index": 1
        }
        
        # Ajouter le résultat à la liste des résultats
        results.append(result)
    
    return calculate_statistics(results)

def calculate_statistics(results):
    diff_distances_euclid = []
    diff_minus_one_distances_euclid = []
    diff_plus_one_distances_euclid = []
    diff_minus_two_distances_euclid = []
    diff_plus_two_distances_euclid = []
    
    diff_distances_cosinus = []
    diff_minus_one_distances_cosinus = []
    diff_plus_one_distances_cosinus = []
    diff_minus_two_distances_cosinus = []
    diff_plus_two_distances_cosinus = []
    
    for result in results:
        distances_euclid = result['distance_euclid']
        distances_cosinus = result['distance_cosinus']
        idx = result['different_word_index']
        
        if idx <= 0 or idx >= len(distances_euclid) - 1:
            continue  # Ignorer les cas où l'indice du mot différent est aux bords ou invalide
        
        # Ajouter la distance du mot différent
        diff_distances_euclid.append(distances_euclid[idx])
        diff_distances_cosinus.append(distances_cosinus[idx])
                
        # Ajouter la distance des mots à un indice de distance, si valides
        if idx - 1 > 0:
            diff_minus_one_distances_euclid.append(distances_euclid[idx - 1])
            diff_minus_one_distances_cosinus.append(distances_cosinus[idx - 1])
        if idx + 1 < len(distances_euclid) - 1:
            diff_plus_one_distances_euclid.append(distances_euclid[idx + 1])
            diff_plus_one_distances_cosinus.append(distances_cosinus[idx + 1])
        
        # Ajouter la distance des mots à deux indices de distance, si valides
        if idx - 2 > 0:
            diff_minus_two_distances_euclid.append(distances_euclid[idx - 2])
            diff_minus_two_distances_cosinus.append(distances_cosinus[idx - 2])
        if idx + 2 < len(distances_euclid) - 1:
            diff_plus_two_distances_euclid.append(distances_euclid[idx + 2])
            diff_plus_two_distances_cosinus.append(distances_cosinus[idx + 2])
    
    # Calculer les moyennes
    mean_diff_distance_euclid = np.mean(diff_distances_euclid) if diff_distances_euclid else float('nan')
    mean_diff_minus_one_distance_euclid = np.mean(diff_minus_one_distances_euclid) if diff_minus_one_distances_euclid else float('nan')
    mean_diff_plus_one_distance_euclid = np.mean(diff_plus_one_distances_euclid) if diff_plus_one_distances_euclid else float('nan')
    mean_diff_minus_two_distance_euclid = np.mean(diff_minus_two_distances_euclid) if diff_minus_two_distances_euclid else float('nan')
    mean_diff_plus_two_distance_euclid = np.mean(diff_plus_two_distances_euclid) if diff_plus_two_distances_euclid else float('nan')
    
    mean_diff_distance_cosinus = np.mean(diff_distances_cosinus) if diff_distances_cosinus else float('nan')
    mean_diff_minus_one_distance_cosinus = np.mean(diff_minus_one_distances_cosinus) if diff_minus_one_distances_cosinus else float('nan')
    mean_diff_plus_one_distance_cosinus = np.mean(diff_plus_one_distances_cosinus) if diff_plus_one_distances_cosinus else float('nan')
    mean_diff_minus_two_distance_cosinus = np.mean(diff_minus_two_distances_cosinus) if diff_minus_two_distances_cosinus else float('nan')
    mean_diff_plus_two_distance_cosinus = np.mean(diff_plus_two_distances_cosinus) if diff_plus_two_distances_cosinus else float('nan')
    
    stat_euclid = {
        "mean_diff_distance_euclid": mean_diff_distance_euclid,
        "mean_diff_minus_one_distance_euclid": mean_diff_minus_one_distance_euclid,
        "mean_diff_plus_one_distance_euclid": mean_diff_plus_one_distance_euclid,
        "mean_diff_minus_two_distance_euclid": mean_diff_minus_two_distance_euclid,
        "mean_diff_plus_two_distance_euclid": mean_diff_plus_two_distance_euclid
    }
    
    stat_cosinus = {
        "mean_diff_distance_cosinus": mean_diff_distance_cosinus,
        "mean_diff_minus_one_distance_cosinus": mean_diff_minus_one_distance_cosinus,
        "mean_diff_plus_one_distance_cosinus": mean_diff_plus_one_distance_cosinus,
        "mean_diff_minus_two_distance_cosinus": mean_diff_minus_two_distance_cosinus,
        "mean_diff_plus_two_distance_cosinus": mean_diff_plus_two_distance_cosinus
    }
    
    return {
        "stat_euclid": stat_euclid,
        "stat_cosinus": stat_cosinus
    }
