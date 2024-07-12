import torch
import random
import itertools
import math
from collections import Counter
import json

def get_embedding(inputs, model):
    with torch.no_grad():
        encoder_outputs = model.model.encoder(**inputs)
        embedding = encoder_outputs.last_hidden_state
    return encoder_outputs, embedding[0][1:-1].cpu().detach().numpy()

def decode_embedding(encoder_outputs, model, tokenizer):
    # Initialiser l'entrée du décodeur avec le token de début de séquence
    decoder_start_token = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token]])
    
    # Générer la séquence de sortie en utilisant les embeddings de l'encodeur
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

#8 pts pour n=2
#Ne pas utiliser
def rotate_around_point_expo(vector, distance):
    n = len(vector)
    rotated_vectors = []

    # Generate all positions
    positions = generate_positions(n, distance)

    # Calculate the rotated vectors
    for pos in positions:
        rotated_vector = [vector[i] + pos[i] for i in range(n)]
        rotated_vectors.append(rotated_vector)

    return rotated_vectors

#4 pts pour n=2
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
def find_neighbor_around(embedding, encoder_outputs, model, tokenizer, neighbor_number=1, step=0.5, start_distance=0, min_lap=0, max_lap=1):
    
    n = 0
    
    if start_distance == 0:
        start_distance = step
    
    if min_lap>max_lap:
        max_lap = min_lap
        
    words = []
    
    while (len(words) <= neighbor_number and n<=min_lap) or max_lap>n:
        words = []
        print("- Distance " + str(start_distance + n*step) + " : ")
        
        list_embeddings = []
        for i in range(len(embedding)):
            embeddings = rotate_around_point_lin(embedding[i], start_distance + n*step)
            list_embeddings.append(embeddings)
            
        for i in range(len(list_embeddings[0])):
            for j in range(len(list_embeddings)):
                encoder_outputs.last_hidden_state[:, 1+j:2+j, :] = torch.FloatTensor(list_embeddings[j][i])
                result = decode_embedding(encoder_outputs, model, tokenizer)
            words.append(result)

        words = dict(Counter(words))
        print(words)
        
        n += 1

    return words

def interpolate_vectors(v1, v2, n):
    vectors = []
    for i in range(n):
        alpha = i / (n - 1)  # interpolation parameter from 0 to 1
        interpolated_vector = (1 - alpha) * v1 + alpha * v2
        vectors.append(interpolated_vector)
    return vectors

def interpolate_test(tokenizer, model, input_text1, input_text2, n=100):
    input_text = "a"
    inputs = tokenizer(input_text, return_tensors="pt")
    encoder_outputs, embedding = get_embedding(inputs, model)

    inputs1 = tokenizer(input_text1, return_tensors="pt")
    encoder_outputs1, embedding1 = get_embedding(inputs1, model)

    inputs2 = tokenizer(input_text2, return_tensors="pt")
    encoder_outputs2, embedding2 = get_embedding(inputs2, model)
    
    assert len(embedding1) == 1 and len(embedding2) == 1

    interpolated_vectors = interpolate_vectors(embedding1, embedding2, n)

    result = []
    result.append(input_text1)
    for e in interpolated_vectors:
        encoder_outputs.last_hidden_state[:, 1:2, :] = torch.FloatTensor(e)
        result.append(decode_embedding(encoder_outputs, model, tokenizer))
        
    result.append(input_text2)
    result = dict(Counter(result))
    return result

def random_interpolate_test(tokenizer, model, nb_result, nb_point=100):
    result = []
    while len(result) < nb_result:
        
        input_text = "a"
        inputs = tokenizer(input_text, return_tensors="pt")
        encoder_outputs, embedding = get_embedding(inputs, model)
        
        input_text1 = random_word()
        inputs1 = tokenizer(input_text1, return_tensors="pt")
        encoder_outputs1, embedding1 = get_embedding(inputs1, model)
        
        while len(embedding1) != 1:
            input_text1 = random_word()
            inputs1 = tokenizer(input_text1, return_tensors="pt")
            encoder_outputs1, embedding1 = get_embedding(inputs1, model)
            
        input_text2 = random_word()
        inputs2 = tokenizer(input_text2, return_tensors="pt")
        encoder_outputs1, embedding2 = get_embedding(inputs2, model)
        
        while len(embedding2) != 1:
            input_text2 = random_word()
            inputs2 = tokenizer(input_text2, return_tensors="pt")
            encoder_outputs1, embedding2 = get_embedding(inputs2, model)
            
        interpolated_vectors = interpolate_vectors(embedding1, embedding2, nb_point)
        
        r = []
        r.append(input_text1)
        for e in interpolated_vectors:
            encoder_outputs.last_hidden_state[:, 1:2, :] = torch.FloatTensor(e)
            r.append(decode_embedding(encoder_outputs, model, tokenizer))
        r.append(input_text2)
         
        r = dict(Counter(r))
        if len(r)>=3:
            result.append(r)
            print(str(len(result)) + "/" + str(nb_result))
    
           
    with open('result.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)    