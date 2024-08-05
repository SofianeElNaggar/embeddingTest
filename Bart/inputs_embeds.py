from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from tools import *
from  input_embedding_tools import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer, model = load_tokenizer_and_model("facebook/bart-large")
model.to(device)

text = "hello"

inputs = tokenizer(text, return_tensors='pt').to(device)

# Obtenir les embeddings d'entrée
input_ids = inputs['input_ids']
input_embeddings = model.get_input_embeddings()
inputs_embeds = input_embeddings(input_ids)



sauvegarder_en_json(find_neighbor_around(model, tokenizer, device, input_embeddings, inputs_embeds, input_ids, min_lap=5),"./Bart/results/new test/neighbor/neighbor_hello.json")


print("Embedding input before : \n" + str(inputs_embeds))

# Générer le texte avant la modification des embeddings
text_before, batch_text_before = decode(model, tokenizer, input_ids, inputs_embeds)
print("Text before : \n" + text_before + "\n" + str(batch_text_before))

# Modifier les embeddings d'entrée (ici, ajouter une petite valeur pour la démonstration)
modified_embeds = inputs_embeds - inputs_embeds

print("\nEmbedding input after : \n" + str(modified_embeds))

set_new_embedding(model, device, input_embeddings, input_ids, modified_embeds)

# Générer le texte après la modification des embeddings
text_after, batch_text_after = decode(model, tokenizer, input_ids, inputs_embeds)
print("Text after : \n" + text_after + "\n" + str(batch_text_after))
