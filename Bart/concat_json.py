import json
import os

def concat_json_files(input_folder, output_file):
    # Initialiser une liste pour stocker le contenu filtré de tous les fichiers JSON
    all_data = []

    # Parcourir tous les fichiers dans le dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Filtrer les objets où le milieu est une chaîne vide ""
                filtered_data = []
                for item in data:
                    # Vérifier chaque clé et sa valeur
                    valid = True
                    for key, value in item.items():
                        if key == "":
                            valid = False
                            break
                    if valid:
                        filtered_data.append(item)
                
                # Ajouter le contenu filtré du fichier à la liste globale
                all_data.extend(filtered_data)

    # Écrire la liste globale filtrée dans le fichier de sortie
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(all_data, outfile, ensure_ascii=False, indent=4)
        
        
# Utilisation de la fonction
input_folder = './Bart/result_interpolation'  # Remplacez par le chemin de votre dossier contenant les fichiers JSON
output_file = './Bart/result_interpolation.json'  # Remplacez par le chemin du fichier de sortie souhaité

concat_json_files(input_folder, output_file)
