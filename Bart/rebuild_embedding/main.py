import torch
from  input_embedding_tools import *
from full_embedding import tools as tools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer, model = tools.load_tokenizer_and_model("facebook/bart-large")
model.to(device)

def modif_embedding():
    text = "hello"
    inputs = tokenizer(text, return_tensors='pt').to(device)
    input_ids, embedding, inputs_embeddings = get_embeddings(inputs, model)
    print(inputs_embeddings)


    print("Embedding input before : \n" + str(inputs_embeddings))

    # Générer le texte avant la modification des embeddings
    text_before = decode(model, tokenizer, input_ids, inputs_embeddings)
    print("Text before : \n" + text_before)

    # Modifier les embeddings d'entrée (ici, ajouter une petite valeur pour la démonstration)
    modified_embeds = inputs_embeddings - inputs_embeddings

    print("\nEmbedding input after : \n" + str(modified_embeds))

    set_new_embedding(model, device, embedding, input_ids, modified_embeds)

    # Générer le texte après la modification des embeddings
    text_after = decode(model, tokenizer, input_ids, inputs_embeddings)
    print("Text after : \n" + text_after)

#print(get_closest_words(calculate_distances(" queen", "./Bart/inputs/bart_all_word_embedding.pkl"),10))

#modif_embedding()
