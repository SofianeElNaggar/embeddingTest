import re
import random
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig
from get_embeding import *
import numpy as np

#Sélection un mot aléatoire dans un fichier
def random_word():
    fichier = './english-common-words.txt'
    with open(fichier, 'r') as f:
        lignes = f.readlines()
        ligne = random.choice(lignes)
    return ligne.strip()

#Tranforme les tokens de sortie en un string
def token_to_string(output_tokens):
    string = ""
    output_tokens = output_tokens[1:-1]
    i = 0
    for t in output_tokens:
        if t.startswith("##"):
            string = string + t[2:]
            i += 1
        else:
            if i == 0:
                string = t
            else:
                string = string + " " + t
            i += 1
    return string

#Compare deux string mot à mot
#Retourne le nombre de différence entre ces deux strings
#Les ponctuations sont considérées comme des mots
def compare_strings(str1, str2):
    # Fonction pour normaliser les chaînes de caractères
    def normalize_string(s):
        # Convertir en minuscules
        s = s.lower()
        # Enlever les espaces avant les ponctuations
        s = re.sub(r'\s+([,.!?:;])', r'\1', s)
        # Remplacer les espaces multiples par un seul espace
        s = re.sub(r'\s+', ' ', s)
        # Diviser en mots, en incluant les ponctuations comme des mots
        words = re.findall(r'\w+|[,.!?:;]', s)
        return words
    
    # Normaliser les deux chaînes de caractères
    words1 = normalize_string(str1)
    words2 = normalize_string(str2)
    
    # Comparer les deux listes de mots
    diff_count = 0
    error = []
    for w1, w2 in zip(words1, words2):
        if w1 != w2:
            diff_count += 1
            error.append([w1,w2])

    # Ajouter les mots restants dans la chaîne la plus longue
    diff_count += abs(len(words1) - len(words2))

    return diff_count, error

#Test le nombre d'erreur entre l'input et l'output en fonction du paramètre hidden_size
#range1 : plage de test
#range2 : nombre de passage
#steps : incrémentation du paramètre hidden_size (doit être un multiple de 12)
def test_hidden_size(input_text, tokenizer, range1=10, range2=1, steps=12):
    
    nb_error_evolv = []

    for j in range(range2):
        for i in range(1,range1):
            configuration = BertConfig(hidden_size=(steps*i))
            model = BertModel(configuration)
            
            embedding = get_bert_embedding(input_text,tokenizer,model)
            output_tokens = iterate_tokens(embedding,tokenizer,model)
            output_text = token_to_string(output_tokens)
            
            nb_error, errors = compare_strings(input_text, output_text)
            
            print("j : " + str(j))
            print("i : " + str(i))
            print("nb erreur : " + str(nb_error))
            
            if j == 0:
                nb_error_evolv.append(nb_error)
            else:
                nb_error_evolv[i-1] += nb_error


    indices = range(1, len(nb_error_evolv) + 1)

    plt.bar(indices, nb_error_evolv)

    plt.xlabel('Indice')
    plt.ylabel('nb error')

    plt.show()

#Test la distance entre deux embeddings de sortie
#Retourne les statistiques du test
#input_text : le mot/text à testé (n'est pas nécéssaire si random est True)
#n : nombre de test à faire
#random : True -> test sur des mots aléatoire a chaque passage (l'embedding de chaque mot est comparé à celui d'un autre)
#         False -> test sur input_text
#h_size : hidden_size paramètre de BertConfig
def test_distance(tokenizer, input_text=None, n=100, random=False, h_size=(12*32)):
    
    distances = []
    last_embedding = None
    
    for i in range(n):
        if random:
            input_text = random_word()
        
        configuration = BertConfig(hidden_size = h_size)
        model = BertModel(configuration)

        embedding = get_bert_embedding(input_text,tokenizer,model)
        embedding = get_embedding_of_word(embedding)
        embedding = embedding.cpu().detach().numpy()[0]
        embedding = np.squeeze(embedding)
        
        if i != 0:
            distance = cosine(embedding, last_embedding)
            distances.append(distance)
        last_embedding = embedding
        
    # Calcul de la moyenne
    moyenne = np.mean(distances)
    print(f'Moyenne: {moyenne}')

    # Calcul de la médiane
    mediane = np.median(distances)
    print(f'Médiane: {mediane}')

    # Calcul des quartiles
    premier_quartile = np.percentile(distances, 25)
    troisieme_quartile = np.percentile(distances, 75)
    print(f'Premier quartile (Q1): {premier_quartile}')
    print(f'Troisième quartile (Q3): {troisieme_quartile}')

    # Affichage du diagramme en boîte à moustaches
    plt.boxplot(distances)
    if random:
        plt.title('word : random_word() - nb itération : ' + str(n) + ' hidden_size = ' + str(h_size))
    else:
        plt.title('word : ' + input_text + ' - nb itération : ' + str(n) + ' hidden_size = ' + str(h_size))
    plt.ylabel('Distance')
    plt.show()

#Pas sur de l'utilité de celui là
def test_nums_hidden_layers(input_text, tokenizer, range1=25, range2=1):
    nb_error_evolv = []

    for j in range(range2):
        for i in range(1,range1):
            configuration = BertConfig(num_hidden_layers=i)
            model = BertModel(configuration)
            
            embedding = get_bert_embedding(input_text,tokenizer,model)
            output_tokens = iterate_tokens(embedding,tokenizer,model)
            output_text = token_to_string(output_tokens)
            
            nb_error, errors = compare_strings(input_text, output_text)
            
            print("j : " + str(j))
            print("i : " + str(i))
            print('output : ' + output_text)
            print("nb erreur : " + str(nb_error))
            
            if j == 0:
                nb_error_evolv.append(nb_error)
            else:
                nb_error_evolv[i-1] += nb_error


    indices = range(1, len(nb_error_evolv) + 1)

    plt.bar(indices, nb_error_evolv)

    plt.xlabel('Indice')
    plt.ylabel('nb error')

    plt.show()
    
def nb_error(tokenizer, model, n):
    nb_error = 0

    for i in range(n):
        print(i)
        input_text = random_word()

        embedding = get_bert_embedding(input_text,tokenizer,model)
        output = iterate_tokens(embedding, tokenizer, model)
        output_word = token_to_string(output)
        
        if input_text != output_word:
            nb_error += 1
        
    return nb_error
