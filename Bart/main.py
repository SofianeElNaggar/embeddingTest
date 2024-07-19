from transformers import BartForConditionalGeneration, BartTokenizer
from tools import *
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter

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


"""
input_text = "A man is <mask> in the street"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)
print("input : " + input_text)
print("output : " + str(batch_decode_embedding(encoder_outputs, model, tokenizer)))
"""

"""
list_of_lists = number_of_dimension_change(model, tokenizer, device, 0.05,"Bart/inputs/sentences_pair.json")

# Aplatir la liste de listes en une seule liste
flattened_list = [item for sublist in list_of_lists for item in sublist]

# Compter les occurrences de chaque valeur
counter = Counter(flattened_list)

# Préparer les données pour l'histogramme
values = list(counter.keys())
frequencies = list(counter.values())

# Créer l'histogramme
plt.figure(figsize=(10, 6))
plt.bar(values, frequencies, width=1.0, edgecolor='black')
plt.xlabel('Valeurs')
plt.ylabel('Fréquence d\'apparition')
plt.title('Histogramme des apparitions des valeurs')
plt.grid(True)
plt.show()
"""

"""
translation_vector = compute_translation_vectors(tokenizer, model, device, "./Bart/inputs/word_pair.json")
print(compute_average_distances(translation_vector))
# ???? Distance cosine de presque 1 ??????? ça devrait être ~0
"""

"""
input_text = "hello"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)

noise = add_noise_to_special_tokens(model, tokenizer, device, encoder_outputs, 100, 0)
sauvegarder_en_json(noise, './Bart/results/noise_last_special_token_hello.json')
"""


"""
c = change_special_tokens_vectors(model, tokenizer, device, embedding, 3000)
sauvegarder_en_json(c, './Bart/results/test_special_token_hello.json')
"""


"""
input_text = "hello"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)

sauvegarder_en_json(find_neighbor_around(embedding, encoder_outputs, model, tokenizer, device, step=0.5, start_distance=0.5, min_lap=10), './Bart/results/neighbor/neighbor_hello_token_limite.json')
"""


"""
map = calculate_distances_and_indices("Bart/inputs/sentences_pair.json", device, tokenizer, model)
stat = calculate_statistics(map)
sauvegarder_en_json(stat, "./Bart/results/stats_distances_inv.json")
"""


"""
input_text2 = "Fonction computing gcd"
inputs2 = tokenizer(input_text2, return_tensors="pt").to(device)
encoder_outputs2, embedding2 = get_embedding(inputs2, model)
e2 = encoder_outputs2.last_hidden_state[0].cpu().detach().numpy()

print(distance_cosinus_between_vectors(e1, e2))
"""

input_text = "Queen"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)
sauvegarder_en_json(change_all_dimension(model, tokenizer, encoder_outputs), "Bart/results/change_all_dim_queen.json")

#sauvegarder_en_json(random_interpolate_test(tokenizer, model, device, 100),"./Bart/interpolation.json")

#print(decode_embedding(encoder_outputs, model, tokenizer))

#sauvegarder_en_json(distance_between_random_words(device, model, tokenizer, 1000), "./Bart/stats_distance_random.json")

def king_queen():
    """
    c'est moche mais c'est vite fait
    les résultats sont nul
    """
    king_text = "king"
    king_inputs = tokenizer(king_text, return_tensors="pt").to(device)
    king_encoder_outputs, king_embedding = get_embedding(king_inputs, model)
    man_text = "man"
    man_inputs = tokenizer(man_text, return_tensors="pt").to(device)
    man_encoder_outputs, man_embedding = get_embedding(man_inputs, model)

    vec_man_king = []
    for i in range(len(king_embedding[0])):
        vec_man_king.append(man_embedding[0][i]-king_embedding[0][i])
        
    queen_text = "queen"
    queen_inputs = tokenizer(queen_text, return_tensors="pt").to(device)
    queen_encoder_outputs, queen_embedding = get_embedding(queen_inputs, model)
    woman_text = "woman"
    woman_inputs = tokenizer(woman_text, return_tensors="pt").to(device)
    woman_encoder_outputs, woman_embedding = get_embedding(woman_inputs, model)

    vec_woman_queen = []
    for i in range(len(queen_embedding[0])):
        vec_woman_queen.append(woman_embedding[0][i]-queen_embedding[0][i])
        
    print(cosinus_distance(vec_man_king, vec_woman_queen))

#king_queen()