import torch
import random
import itertools
import math

def get_embedding(inputs, model):
    with torch.no_grad():
        encoder_outputs = model.model.encoder(**inputs)
        embedding = encoder_outputs.last_hidden_state
    return encoder_outputs, embedding[:,1:2, :].cpu().detach().numpy()[0][0]

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

#8 pts pour n=2
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