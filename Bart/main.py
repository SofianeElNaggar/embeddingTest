from transformers import BartForConditionalGeneration, BartTokenizer
from tools import *
import numpy as np
import json

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = "facebook/bart-large"

tokenizer = BartTokenizer.from_pretrained(model)
model = BartForConditionalGeneration.from_pretrained(model)

model.to(device)

input_text = "hello"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)

sauvegarder_en_json(find_neighbor_around(embedding, encoder_outputs, model, tokenizer, device, step=0.5, start_distance=0.5, max_lap=1), './Bart/neighbor_hello_05.json')


"""map = calculate_distances_and_indices("Bart/sentences_pair.json", device, tokenizer, model)
stat = calculate_statistics(map)

sauvegarder_en_json(stat, "./Bart/stats_distances.json")"""



"""
input_text2 = "Fonction computing gcd"
inputs2 = tokenizer(input_text2, return_tensors="pt").to(device)
encoder_outputs2, embedding2 = get_embedding(inputs2, model)
e2 = encoder_outputs2.last_hidden_state[0].cpu().detach().numpy()"""


#print(distance_cosinus_between_vectors(e1, e2))

#random_interpolate_test(tokenizer, model, device, 50)

#print(decode_embedding(encoder_outputs, model, tokenizer))



#sauvegarder_en_json(distance_between_random_words(device, model, tokenizer, 1000), "./Bart/stats_distance_random.json")
