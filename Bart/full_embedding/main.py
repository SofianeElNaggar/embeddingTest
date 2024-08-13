from tools import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "facebook/bart-large"

tokenizer, model = load_tokenizer_and_model(model_name)

model.to(device)

#print(model.config)

def s_sentences():
    text = "He is a poor boy. You have no sympathy. You are my friend."
    perturb_text = " We have no sympathy."
    switch_sentences(text, perturb_text)

def noise_special_tokens():
    input_text = "hello"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    encoder_outputs, embedding = get_embedding(inputs, model)

    noise = add_noise_to_special_tokens(model, tokenizer, device, encoder_outputs, 100, 0)
    sauvegarder_en_json(noise, './Bart/results/noise_last_special_token_hello.json')

def change_special_tokens(n = 100):
    input_text = "hello"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    encoder_outputs, embedding = get_embedding(inputs, model)
    c = change_special_tokens_vectors(model, tokenizer, device, embedding, n)
    sauvegarder_en_json(c, './Bart/results/test_special_token_hello.json')

def neighbor_random_word():
    input_text = random_word()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    encoder_outputs, embedding = get_embedding(inputs, model)

    sauvegarder_en_json(find_neighbor_around(embedding, encoder_outputs, model, tokenizer, device, step=0.5, start_distance=0.5, min_lap=10), './Bart/results/neighbor/neighbor_' + input_text +'_token_limit.json')

def change_all_dim():
    input_text = "model"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    encoder_outputs, embedding = get_embedding(inputs, model)
    sauvegarder_en_json(change_all_dimension(model, tokenizer, encoder_outputs), "Bart/results/results token limit/change dim/change_all_dim_model.json")


#s_sentences()

#graph_change_dim(model, tokenizer, device)

#print(compute_average_distances(compute_translation_vectors(tokenizer, model, device, "./Bart/inputs/word_pair.json")))

#noise_special_tokens()

#change_special_tokens(500)

#neighbor_random_word()

#print(calculate_statistics(calculate_distances_and_indices("./Bart/inputs/sentences_pair.json", device, tokenizer, model)))

#change_all_dim()

#sauvegarder_en_json(random_interpolate_test(tokenizer, model, device, 100),"./Bart/results/results token limit/interpolation2.json")

#sauvegarder_en_json(distance_between_random_words(device, model, tokenizer, 500), "./Bart/stats_distance_random.json")

