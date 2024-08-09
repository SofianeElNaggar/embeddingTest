from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import tools as tools
from collections import Counter
import numpy as np
from scipy.spatial.distance import euclidean
import pickle

def decode(model, tokenizer, input_ids, inputs_embeddings):
    """
    Generates text based on the input_ids using the provided model and tokenizer.

    Parameters:
    - model: The language model used to generate text.
    - tokenizer: The tokenizer to decode the generated output.
    - input_ids: Tensor containing token ids of the input sequence.
    - inputs_embeddings: Tensor containing input embeddings used to limit the length of generated text.

    Returns:
    - text: The decoded text generated by the model.
    """
    outputs = model.generate(input_ids, max_new_tokens=len(inputs_embeddings[0]))
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def set_new_embedding(model, device, embedding, input_ids, modified_embeds):
    """
    Replaces the model's input embeddings with modified embeddings for specific tokens.

    Parameters:
    - model: The language model whose embeddings will be modified.
    - device: The device (e.g., 'cpu' or 'cuda') to which tensors are moved.
    - embedding: The original embedding layer of the model.
    - input_ids: Tensor containing token ids for which embeddings will be modified.
    - modified_embeds: Tensor containing the new embeddings to replace the old ones.

    Returns:
    - None: The function updates the model's embedding layer in place.
    """
    vocab_size, embedding_dim = embedding.weight.shape
    new_embeddings = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
    new_embeddings.weight.data.copy_(embedding.weight.data)
    new_embeddings.weight.data[input_ids] = modified_embeds.squeeze(0).to(device)

    model.set_input_embeddings(new_embeddings)

def get_embeddings(inputs, model):
    """
    Retrieves the input ids and their corresponding embeddings from the model.

    Parameters:
    - inputs: Dictionary containing tokenized inputs with 'input_ids' as a key.
    - model: The language model used to obtain the embeddings.

    Returns:
    - input_ids: Tensor containing token ids from the inputs.
    - embedding: The embedding layer of the model.
    - inputs_embeddings: Tensor containing the embeddings corresponding to the input ids.
    """
    input_ids = inputs['input_ids']
    embedding = model.get_input_embeddings()
    inputs_embeddings = embedding(input_ids)
    return input_ids, embedding, inputs_embeddings

def process_file(model, tokenizer, device, file_path):
    """
    Processes a text file to generate a mapping of words to their embeddings.

    Parameters:
    - model: The language model used to generate embeddings.
    - tokenizer: The tokenizer used to convert words into token ids.
    - device: The device (e.g., 'cpu' or 'cuda') to which tensors are moved.
    - file_path: Path to the text file containing words to be processed.

    Returns:
    - map: A dictionary mapping words to their respective embeddings.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    map = {}
    i = 0
    for line in lines:
        print(str(i) + "/50265")
        i += 1
        word = line.split(':')[0].strip()
        word = word.replace('Ġ', ' ')  # Special token handling

        inputs = tokenizer(word, return_tensors='pt').to(device)

        input_ids = inputs['input_ids']
        embedding = model.get_input_embeddings()
        inputs_embeddings = embedding(input_ids)
        embedding = inputs_embeddings.cpu().detach().numpy()
        if len(embedding[0]) > 3:  # Filter out embeddings with more than 3 dimensions (like : "Ļ")
            continue
        map[word] = embedding[0][1]
        
    return map

def get_vector(word, file_path):
    """
    Retrieves the embedding vector of a specific word from a file.

    Parameters:
    - word: The word for which the embedding vector is to be retrieved.
    - file_path: Path to the file containing the word embeddings.

    Returns:
    - vector: The embedding vector for the specified word, or None if not found.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    return data.get(word, None)

def calculate_distances(target_word, file_path):
    """
    Calculates the Euclidean distance between a target word's embedding and all other words' embeddings in a file.

    Parameters:
    - target_word: The word for which distances to other words are calculated.
    - file_path: Path to the file containing word embeddings.

    Returns:
    - distances: A dictionary mapping each word to its Euclidean distance from the target word.
    """
    target_vector = get_vector(target_word, file_path)
    
    if target_vector is None:
        raise ValueError(f"The word '{target_word}' was not found in the file.")
    
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    distances = {}
    for word, vector in data.items():
        if word != target_word:
            distance = euclidean(target_vector, vector)
            distances[word] = distance
    
    return distances

def get_closest_words(distances, n):
    """
    Retrieves the n closest words to a target word based on precomputed distances.

    Parameters:
    - distances: A dictionary mapping words to their Euclidean distances from a target word.
    - n: The number of closest words to return.

    Returns:
    - closest_words: A list of the n closest words.
    """
    closest_words = sorted(distances.items(), key=lambda item: item[1])[:n]
    return [word for word, distance in closest_words]
