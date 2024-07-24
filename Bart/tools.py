import torch
import random
import itertools
import math
from collections import Counter
import json
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import euclidean

def get_embedding(inputs, model):
    """
    Extracts embeddings from the model's encoder for given inputs.
    
    Parameters:
    inputs (dict): Input tensor dictionary compatible with the model.
    model (transformers.PreTrainedModel): The transformer model used for encoding.
    
    Returns:
    tuple: Encoder outputs and the extracted embeddings as a numpy array.
    """
    with torch.no_grad():
        encoder_outputs = model.model.encoder(**inputs)
        embedding = encoder_outputs.last_hidden_state
    return encoder_outputs, embedding[0][1:-1].cpu().detach().numpy()

def batch_decode_embedding(encoder_outputs, model, tokenizer):
    """
    Decodes the embeddings into human-readable text using the model's decoder.
    
    Parameters:
    encoder_outputs (transformers.modeling_outputs.BaseModelOutput): Outputs from the model's encoder.
    model (transformers.PreTrainedModel): The transformer model used for decoding.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to decode the output tokens.
    
    Returns:
    list: Decoded sequences as tokenized text.
    """
    decoder_start_token = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token]], device=encoder_outputs.last_hidden_state.device)
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=len(encoder_outputs.last_hidden_state[0]),
            num_beams=1,
            early_stopping=False
        )
    
    result = tokenizer.batch_decode(output_sequences[0])
    
    return result

def decode_embedding(encoder_outputs, model, tokenizer):
    """
    Decodes the embeddings into human-readable text using the model's decoder.
    
    Parameters:
    encoder_outputs (transformers.modeling_outputs.BaseModelOutput): Outputs from the model's encoder.
    model (transformers.PreTrainedModel): The transformer model used for decoding.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to decode the output tokens.
    
    Returns:
    str: Decoded sequence as text.
    """
    decoder_start_token = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token]], device=encoder_outputs.last_hidden_state.device)
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            num_beams=1,
            early_stopping=False,
            max_new_tokens=len(encoder_outputs.last_hidden_state[0])
        )
    
    result = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    return result

def random_word():
    """
    Selects a random word from a file of common English words.
    
    Returns:
    str: A randomly selected word.
    """
    fichier = './english-common-words.txt'
    with open(fichier, 'r') as f:
        lignes = f.readlines()
        ligne = random.choice(lignes)
    return ligne.strip()

def generate_positions(n, distance):
    """
    Generates positions in n-dimensional space around a point at a given distance.
    
    Parameters:
    n (int): Number of dimensions.
    distance (float): Distance from the origin.
    
    Returns:
    list: List of positions around the origin.
    """
    directions = list(itertools.product([-1, 0, 1], repeat=n))
    directions = [d for d in directions if any(d)]
    
    positions = []
    for direction in directions:
        length = math.sqrt(sum(d**2 for d in direction))
        normalized_direction = [d * distance / length for d in direction]
        positions.append(normalized_direction)
    
    return positions

def rotate_around_point_lin(vector, distance):
    """
    Generates rotated vectors around a given vector in n-dimensional space.
    
    Parameters:
    vector (list): The original vector.
    distance (float): Distance to rotate around the original vector.
    
    Returns:
    list: List of rotated vectors.
    """
    n = len(vector)
    rotated_vectors = []

    positions = []
    for i in range(n):
        pos = [0] * n
        pos[i] = distance
        positions.append(pos)
        pos = [0] * n
        pos[i] = -distance
        positions.append(pos)

    if n == 2:
        diagonal_distance = distance / math.sqrt(2)
        positions.extend([
            [diagonal_distance, diagonal_distance],
            [-diagonal_distance, diagonal_distance],
            [diagonal_distance, -diagonal_distance],
            [-diagonal_distance, -diagonal_distance]
        ])

    for pos in positions:
        rotated_vector = [vector[i] + pos[i] for i in range(n)]
        rotated_vectors.append(rotated_vector)

    return rotated_vectors

def find_neighbor_around(embedding, encoder_outputs, model, tokenizer, device, neighbor_number=1, step=0.5, start_distance=0, min_lap=0, max_lap=1):
    """
    Finds neighbors around a given embedding by rotating in n-dimensional space.
    
    Parameters:
    embedding (list): The original embedding.
    encoder_outputs (transformers.modeling_outputs.BaseModelOutput): Outputs from the model's encoder.
    model (transformers.PreTrainedModel): The transformer model used for decoding.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to decode the output tokens.
    device (torch.device): Device to perform computations on.
    neighbor_number (int): Number of minimum neighbors to find.
    step (float): Incremental step distance for each iteration.
    start_distance (float): Starting distance for the first iteration.
    min_lap (int): Minimum number of iterations.
    max_lap (int): Maximum number of iterations.
    
    Returns:
    dict: Dictionary of found neighbors with distances as keys.
    """
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
            
        words["Distance : " + str(distance)] = dict(Counter(results))
        distance += step
        n += 1

    return words

def interpolate_vectors(v1, v2, n):
    """
    Interpolates between two vectors linearly.
    
    Parameters:
    v1 (list): The first vector.
    v2 (list): The second vector.
    n (int): Number of interpolation steps.
    
    Returns:
    list: List of interpolated vectors.
    """
    vectors = []
    for i in range(n):
        alpha = i / (n - 1)
        interpolated_vector = (1 - alpha) * v1 + alpha * v2
        vectors.append(interpolated_vector)
    return vectors

def interpolate_test(tokenizer, model, device, input_text1, input_text2, n=100):
    """
    Interpolates between the embeddings of two input texts and decodes the interpolations.
    
    Parameters:
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to encode the input texts.
    model (transformers.PreTrainedModel): The transformer model used for encoding and decoding.
    device (torch.device): Device to perform computations on.
    input_text1 (str): The first input text.
    input_text2 (str): The second input text.
    n (int): Number of interpolation steps.
    
    Returns:
    dict: Counter dictionary of decoded interpolated texts.
    """
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
    """
    Randomly selects word pairs, interpolates between their embeddings, and decodes the results.
    
    Parameters:
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to encode the input texts.
    model (transformers.PreTrainedModel): The transformer model used for encoding and decoding.
    device (torch.device): Device to perform computations on.
    nb_result (int): Number of word pairs to process.
    nb_point (int): Number of interpolation steps.
    
    Returns:
    list: List of dictionaries containing the interpolation results for each pair.
    """
    all_results = []
    
    for i in range(nb_result):
        word1 = random_word()
        word2 = random_word()
        while word1 not in tokenizer.get_vocab() or word2 not in tokenizer.get_vocab():
            word1 = random_word()
            word2 = random_word()
        result = interpolate_test(tokenizer, model, device, word1, word2, n=nb_point)
        all_results.append(result)
    return all_results

def distance_euclid_between_vectors(liste_vecteurs1, liste_vecteurs2):
    """
    Calculate the Euclidean distance between corresponding vectors in two lists.
    
    Parameters:
    liste_vecteurs1 (list): First list of vectors.
    liste_vecteurs2 (list): Second list of vectors.
    
    Returns:
    list: Euclidean distances between each pair of vectors.
    """
    distances = []
    for v1, v2 in zip(liste_vecteurs1, liste_vecteurs2):
        # Convert lists to numpy arrays for easier calculations
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        # Calculate the Euclidean distance between the two vectors
        distance = np.linalg.norm(v1_np - v2_np)
        distances.append(distance)
    return distances

def cosinus_distance(v1, v2):
    """
    Calculate the cosine distance between two vectors.
    
    Parameters:
    v1 (list): First vector.
    v2 (list): Second vector.
    
    Returns:
    float: Cosine distance between the two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_sim = dot_product / (norm_v1 * norm_v2)
    cos_dist = 1 - cos_sim
    return cos_dist

def distance_cosinus_between_vectors(liste_vecteurs1, liste_vecteurs2):
    """
    Calculate the cosine distances between corresponding vectors in two lists.
    
    Parameters:
    liste_vecteurs1 (list): First list of vectors.
    liste_vecteurs2 (list): Second list of vectors.
    
    Returns:
    list: Cosine distances between each pair of vectors.
    """
    if len(liste_vecteurs1) != len(liste_vecteurs2):
        raise ValueError("The two lists of vectors must be of the same length.")
    
    distances = []
    for v1, v2 in zip(liste_vecteurs1, liste_vecteurs2):
        distance = cosinus_distance(v1, v2)
        distances.append(distance)
    
    return distances

def check_tokenization_integrity(sentence, tokenized_sentence):
    """
    Check if the tokenized sentence matches the original sentence in terms of word structure.
    
    Parameters:
    sentence (str): The original sentence.
    tokenized_sentence (list): List of tokens from the tokenized sentence.
    
    Returns:
    bool: True if the tokenized sentence matches the original sentence, False otherwise.
    """
    original_words = sentence.split()
    
    # Remove special tokens if present
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
    
    # Add the last reconstructed word
    if current_word:
        reconstructed_words.append(current_word.strip())
    
    if len(original_words) != len(reconstructed_words):
        return False
    
    for orig_word, recon_word in zip(original_words, reconstructed_words):
        if orig_word != recon_word:
            return False
    
    return True

def find_different_word_index(sentence1, sentence2):
    """
    Find the index of the first different word between two sentences.
    
    Parameters:
    sentence1 (str): The first sentence.
    sentence2 (str): The second sentence.
    
    Returns:
    int: The index of the first different word. Returns -1 if no difference is found.
    """
    words1 = sentence1.split()
    words2 = sentence2.split()
    
    for i in range(len(words1)):
        if words1[i] != words2[i]:
            return i  # Return the index of the first different word
    
    return -1

def calculate_distances_and_indices(json_file, device, tokenizer, model):
    """
    Calculate Euclidean and cosine distances between embeddings of sentence pairs from a JSON file.
    Also find the index of the first different word between each pair.
    
    Parameters:
    json_file (str): Path to the JSON file containing sentence pairs.
    device (torch.device): Device to perform computations on.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding sentences.
    model (transformers.PreTrainedModel): Transformer model for obtaining embeddings.
    
    Returns:
    list: Results containing distances and different word indices for each sentence pair.
    """
    results = []
    
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for pair in data:
        sentence1 = pair['sentence1']
        sentence2 = pair['sentence2']
        
        # Get embeddings for sentence1
        inputs1 = tokenizer(sentence1, return_tensors="pt").to(device)
        encoder_outputs1, embedding1 = get_embedding(inputs1, model)
        e1 = encoder_outputs1.last_hidden_state[0].cpu().detach().numpy()
        
        # Get embeddings for sentence2
        inputs2 = tokenizer(sentence2, return_tensors="pt").to(device)
        encoder_outputs2, embedding2 = get_embedding(inputs2, model)
        e2 = encoder_outputs2.last_hidden_state[0].cpu().detach().numpy()
        
        # Check tokenization integrity
        b1 = check_tokenization_integrity(sentence1, batch_decode_embedding(encoder_outputs1, model, tokenizer))
        b2 = check_tokenization_integrity(sentence2, batch_decode_embedding(encoder_outputs2, model, tokenizer))
        
        if not b1 or not b2:
            continue
        
        if len(e1) != len(e2):
            continue
        
        # Calculate distances between embeddings
        distances_euclid = distance_euclid_between_vectors(e1, e2)
        distances_cosinus = distance_cosinus_between_vectors(e1, e2)
        
        # Find the index of the first different word
        idx_different_word = find_different_word_index(sentence1, sentence2)
        
        # Create a result for this pair
        result = {
            "distance_euclid": distances_euclid,
            "distance_cosinus": distances_cosinus,
            "different_word_index": idx_different_word + 1
        }
        
        # Add the result to the list of results
        results.append(result)
    
    return results

def distance_between_random_words(device, model, tokenizer, n):
    """
    Calculate distances between random word pairs using their embeddings.
    
    Parameters:
    device (torch.device): Device to perform computations on.
    model (transformers.PreTrainedModel): Transformer model for obtaining embeddings.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding words.
    n (int): Number of word pairs to process.
    
    Returns:
    dict: Statistical results of the distances between word pairs.
    """
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
        
        if len(e1) != 3 or len(e2) != 3:
            continue
        
        # Calculate distances between embeddings
        distances_euclid = distance_euclid_between_vectors(e1, e2)
        distances_cosinus = distance_cosinus_between_vectors(e1, e2)
        
        # Create a result for this pair
        result = {
            "distance_euclid": distances_euclid,
            "distance_cosinus": distances_cosinus,
            "different_word_index": 1
        }
        
        # Add the result to the list of results
        results.append(result)
    
    return calculate_statistics(results)

def calculate_statistics(results):
    """
    Calculate statistical measures on the distances between words.
    
    Parameters:
    results (list): List of results containing distances and different word indices.
    
    Returns:
    dict: Statistical measures on Euclidean and cosine distances.
    """
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
            continue  # Ignore cases where the different word index is at the edges or invalid
        
        # Add the distance of the different word
        diff_distances_euclid.append(distances_euclid[idx])
        diff_distances_cosinus.append(distances_cosinus[idx])
                
        # Add the distance of words at one index away, if valid
        if idx - 1 > 0:
            diff_minus_one_distances_euclid.append(distances_euclid[idx - 1])
            diff_minus_one_distances_cosinus.append(distances_cosinus[idx - 1])
        if idx + 1 < len(distances_euclid) - 1:
            diff_plus_one_distances_euclid.append(distances_euclid[idx + 1])
            diff_plus_one_distances_cosinus.append(distances_cosinus[idx + 1])
        
        # Add the distance of words at two indices away, if valid
        if idx - 2 > 0:
            diff_minus_two_distances_euclid.append(distances_euclid[idx - 2])
            diff_minus_two_distances_cosinus.append(distances_cosinus[idx - 2])
        if idx + 2 < len(distances_euclid) - 1:
            diff_plus_two_distances_euclid.append(distances_euclid[idx + 2])
            diff_plus_two_distances_cosinus.append(distances_cosinus[idx + 2])
    
    # Calculate means
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

def dimention_change(v1, v2):
    result = []
    for i in range(len(v1)):
        result.append(LA.norm(v1[i]-v2[i]))
    return result

def number_of_dimension_change(model, tokenizer, device, tol, json_file):
    results = []
    
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for pair in data:
        sentence1 = pair['sentence1']
        sentence2 = pair['sentence2']
        
        # Get embeddings for sentence1
        inputs1 = tokenizer(sentence1, return_tensors="pt").to(device)
        encoder_outputs1, embedding1 = get_embedding(inputs1, model)
        e1 = encoder_outputs1.last_hidden_state[0].cpu().detach().numpy()
        
        # Get embeddings for sentence2
        inputs2 = tokenizer(sentence2, return_tensors="pt").to(device)
        encoder_outputs2, embedding2 = get_embedding(inputs2, model)
        e2 = encoder_outputs2.last_hidden_state[0].cpu().detach().numpy()
        
        # Check tokenization integrity
        b1 = check_tokenization_integrity(sentence1, batch_decode_embedding(encoder_outputs1, model, tokenizer))
        b2 = check_tokenization_integrity(sentence2, batch_decode_embedding(encoder_outputs2, model, tokenizer))
        
        if not b1 or not b2:
            continue
        
        if len(e1) != len(e2):
            continue
        
        # Find the index of the first different word
        idx_different_word = find_different_word_index(sentence1, sentence2)
        
        dif = dimention_change(e1[idx_different_word], e2[idx_different_word])
        i = 0
        v = []        
        for d in dif:
            if d > tol:
                v.append(i)
            i += 1
        results.append(v)
    return results         
        
def change_special_tokens_vectors(model, tokenizer, device, embedding_test, n):
    """
    Replace embeddings of special tokens associated with a word in the model's tokenizer vocabulary 
    with the provided new embedding.

    Parameters:
    model (transformers.PreTrainedModel): Transformer model to modify.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the model.
    device (torch.device): Device to perform computations on.
    embedding_test (list): New embedding vector to replace the special token embeddings.
    n (int): Number of random words to process.

    Returns:
    dict: Mapping of each random word to its new decoded embedding.
    """
    result = {}
    
    for i in range(n):
        random_input = random_word()
        while random_input not in tokenizer.get_vocab():
            random_input = random_word()
        
        inputs = tokenizer(random_input, return_tensors="pt").to(device)
        encoder_outputs, embedding = get_embedding(inputs, model)
        encoder_outputs.last_hidden_state[:, 1:2, :] = torch.FloatTensor(embedding_test).to(device)
        
        result[random_input] = decode_embedding(encoder_outputs, model, tokenizer)
    
    return result

def add_noise(lst, noise_percentage):
    """
    Add random noise to each value in the list, modifying each value by ±10% of its value.

    Parameters:
    lst (list of float/int): The original list of values to which noise will be added.
    noise_percentage (float/int): The percentage of noise to apply.

    Returns:
    list of float: A new list with each original value modified by a random factor between -10% and +10%.
    """
    noisy_list = []
    for value in lst:
        factor = 1 + random.uniform(-noise_percentage/100, noise_percentage/100)
        noisy_value = value * factor
        noisy_list.append(noisy_value)
    return noisy_list

def add_noise_to_special_tokens(model, tokenizer, device, encoder_outputs, n = 100, noise_percentage = 10):
    """
    Add noise to the embeddings of special tokens in the encoder outputs and decode the noisy embeddings.

    Parameters:
    model (transformers.PreTrainedModel): Transformer model used for decoding.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the model.
    device (torch.device): Device to perform computations on.
    encoder_outputs (torch.Tensor): Encoder outputs containing the embeddings.
    n (int, optional): Number of noisy embeddings to generate. Default is 100.
    noise_percentage (float, optional): Percentage of noise to add to the embeddings. Default is 10.

    Returns:
    list: List of decoded embeddings after noise has been added.
    """
    embedding = encoder_outputs.last_hidden_state[0].cpu().detach().numpy()
    result = []
    for j in range(n):
        i = 0
        for e in embedding:
            #noise in the first token as no impact i != 0 or 
            if i != len(embedding)-1:
                i +=1
                continue
            noisy_e = add_noise(e, noise_percentage)
            encoder_outputs.last_hidden_state[0][i] = torch.FloatTensor(noisy_e).to(device)
            i +=1
        result.append(decode_embedding(encoder_outputs, model, tokenizer))
    return result

def compute_translation_vectors(tokenizer, model, device, json_file):
    """
    Computes translation vectors between the embeddings of two words.

    Arguments:
    json_file -- path to a JSON file containing a list of words pairs
                 JSON file format:
                 [
                     {
                         "word1": "king",
                         "word2": "queen"
                     },
                     ...
                 ]

    Returns:
    A list of translation vectors between the embeddings of words.
    Each translation vector is represented as a list of floats.
    """
    with open(json_file, 'r') as file:
        word_pairs = json.load(file)
    
    translation_vectors = []
    
    for pair in word_pairs:
        word1 = pair["word1"]
        word2 = pair["word2"]
        
        if word1 not in tokenizer.get_vocab() or word2 not in tokenizer.get_vocab():
            continue
        
        # Tokenize the words
        word1_inputs = tokenizer(word1, return_tensors="pt").to(device)
        word2_inputs = tokenizer(word2, return_tensors="pt").to(device)
        
        # Get the embeddings
        word1_encoder_outputs, word1_embedding = get_embedding(word1_inputs, model)
        word2_encoder_outputs, word2_embedding = get_embedding(word2_inputs, model)
        
        # Compute the translation vector
        translation_vector = word2_embedding - word1_embedding
        translation_vectors.append(translation_vector.squeeze().tolist())
    
    return translation_vectors

def compute_average_distances(vectors):
    """
    Computes the average Euclidean and cosine distances between each pair of vectors.

    Arguments:
    vectors -- a list of vectors, where each vector is represented as a list of floats.

    Returns:
    A dictionary with the average Euclidean distance and the average cosine distance.
    """
    num_vectors = len(vectors)
    euclidean_distances = []
    cosine_distances = []
    
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            vec1 = np.array(vectors[i])
            vec2 = np.array(vectors[j])
            
            # Compute Euclidean distance
            euclidean_distances.append(euclidean(vec1, vec2))
            
            # Compute cosine distance
            cosine_distances.append(cosinus_distance(vec1, vec2))
    
    average_euclidean = np.mean(euclidean_distances)
    average_cosine = np.mean(cosine_distances)
    
    return {
        "average_euclidean_distance": average_euclidean,
        "average_cosine_distance": average_cosine
    }
    
def change_dimension(model, tokenizer, encoder_outputs, d):
    results = []
    for i in np.arange(-2.5, 2.5, 0.1):
        encoder_outputs.last_hidden_state[0][0][d] = i
        result = decode_embedding(encoder_outputs, model, tokenizer)
        results.append(result)
    return results

def change_all_dimension(model, tokenizer, encoder_outputs):
    results = {}
    for i in range(len(encoder_outputs.last_hidden_state[0][0])):
        result = change_dimension(model, tokenizer, encoder_outputs, i)
        results["dimension : " + str(i)] = remove_consecutive_duplicates(result)
        print(str(i+1) + "/1024")
    return results

def remove_consecutive_duplicates(lst):
    if not lst:
        return []

    result = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            result.append(lst[i])
    return result

def convert_numpy_types(donnees):
    """
    Convertit les types de données NumPy en types natifs Python.
    
    :param donnees: Les données à convertir.
    :return: Les données avec les types convertis.
    """
    if isinstance(donnees, np.ndarray):
        return donnees.tolist()
    elif isinstance(donnees, np.generic):
        return donnees.item()
    elif isinstance(donnees, dict):
        return {k: convert_numpy_types(v) for k, v in donnees.items()}
    elif isinstance(donnees, list):
        return [convert_numpy_types(v) for v in donnees]
    elif isinstance(donnees, tuple):
        return tuple(convert_numpy_types(v) for v in donnees)
    elif isinstance(donnees, set):
        return {convert_numpy_types(v) for v in donnees}
    return donnees

def sauvegarder_en_json(donnees, nom_fichier):
    """
    Sauvegarde les données fournies dans un fichier JSON après conversion des types NumPy.

    :param donnees: Les données à sauvegarder (peut être une liste, un dictionnaire, etc.).
    :param nom_fichier: Le nom du fichier dans lequel sauvegarder les données.
    """
    donnees_converties = convert_numpy_types(donnees)
    
    try:
        with open(nom_fichier, 'w', encoding='utf-8') as fichier:
            json.dump(donnees_converties, fichier, ensure_ascii=False, indent=4)
        print(f"Les données ont été sauvegardées dans le fichier {nom_fichier}.")
    except Exception as e:
        print(f"Une erreur est survenue lors de la sauvegarde : {e}")

