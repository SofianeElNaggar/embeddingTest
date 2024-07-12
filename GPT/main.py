import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# Charger le modèle GPT-2 et le tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Fonction pour obtenir l'embedding d'un mot
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    # Prendre le premier token non spécial (le mot lui-même)
    word_embedding = last_hidden_states
    return word_embedding

# Fonction pour retrouver le mot original à partir de l'embedding
def find_original_word(embedding):
    with torch.no_grad():
        logits = lm_model.lm_head(embedding)
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_words = tokenizer.decode(predicted_ids[0])
    return predicted_words

# Exemple d'utilisation
word = "test"
embedding = get_word_embedding(word)
print(f"Embedding for '{word}':\n{embedding}")

reconstructed_word = find_original_word(embedding)
print(f"Reconstructed word from embedding: {reconstructed_word}")

# Vérifier si l'embedding est vide
if embedding.size(0) == 0:
    print(f"Error: Embedding for '{word}' is empty.")
else:
    reconstructed_word = find_original_word(embedding)
    print(f"Reconstructed word from embedding: {reconstructed_word}")
