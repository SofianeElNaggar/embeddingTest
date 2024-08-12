from tools import *
import torch
import matplotlib.pyplot as plt
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "facebook/bart-large"

tokenizer, model = load_tokenizer_and_model(model_name)

model.to(device)

#print(model.config)

"""
Queen_text = "He is a poor boy. You have no sympathy. You are my friend."

tokenizer, model = tools.load_tokenizer_and_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

Queen_inputs = tokenizer(Queen_text, return_tensors="pt").to(device)
Queen_encoder_outputs, Queen_embedding = tools.get_embedding(Queen_inputs, model)

print("Batch decoding function")
decoded_seq = tools.batch_decode_embedding(Queen_encoder_outputs, model, tokenizer)
print(decoded_seq)
# print(decoded_seq["last_hidden_state"].size())

print("TRY REPLACE an I with You!!!!")
perturb_text = " We have no sympathy."
perturb_inputs = tokenizer(perturb_text, return_tensors="pt").to(device)
perturb_encoder_outputs, perturb_embedding = tools.get_embedding(perturb_inputs, model)

print("Try perturb")
init= Queen_encoder_outputs.last_hidden_state.clone().detach()

#print(init[0,12:-1,:])
#print(perturb_encoder_outputs.last_hidden_state)
Queen_encoder_outputs.last_hidden_state[0,7,:] = perturb_encoder_outputs.last_hidden_state[0,1,:]
#print(Queen_encoder_outputs.last_hidden_state[0,12:-1,:])

print("Try decoding")
perturb_decoded_seq = tools.batch_decode_embedding(Queen_encoder_outputs, model, tokenizer)
print(perturb_decoded_seq)
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
input_text = random_word()
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)

sauvegarder_en_json(find_neighbor_around(embedding, encoder_outputs, model, tokenizer, device, step=0.5, start_distance=0.5, min_lap=10), './Bart/results/results token limit/neighbor/neighbor_' + input_text +'_token_limit.json')

input_text = random_word()
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)

sauvegarder_en_json(find_neighbor_around(embedding, encoder_outputs, model, tokenizer, device, step=0.5, start_distance=0.5, min_lap=10), './Bart/results/results token limit/neighbor/neighbor_' + input_text +'_token_limit.json')

input_text = random_word()
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)

sauvegarder_en_json(find_neighbor_around(embedding, encoder_outputs, model, tokenizer, device, step=0.5, start_distance=0.5, min_lap=10), './Bart/results/results token limit/neighbor/neighbor_' + input_text +'_token_limit.json')
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

"""
input_text = "model"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)
sauvegarder_en_json(change_all_dimension(model, tokenizer, encoder_outputs), "Bart/results/results token limit/change dim/change_all_dim_model.json")
"""

#sauvegarder_en_json(random_interpolate_test(tokenizer, model, device, 1000),"./Bart/results/results token limit/interpolation2.json")

#print(decode_embedding(encoder_outputs, model, tokenizer))

#sauvegarder_en_json(distance_between_random_words(device, model, tokenizer, 1000), "./Bart/stats_distance_random.json")
