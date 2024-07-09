import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

# Charger le tokenizer et le modèle BART pré-entraîné
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Exemple de texte d'entrée
input_text = "Hello"

# Tokeniser le texte d'entrée
inputs = tokenizer(input_text, return_tensors="pt")

# Passer par l'encodeur pour obtenir les embeddings
with torch.no_grad():
    encoder_outputs = model.model.encoder(**inputs)
    embedding = encoder_outputs.last_hidden_state

print(embedding)

# Initialiser l'entrée du décodeur avec le token de début de séquence
decoder_start_token = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token]])

# Générer la séquence de sortie en utilisant les embeddings de l'encodeur
output_sequences = model.generate(
    input_ids=None,
    encoder_outputs=encoder_outputs,
    decoder_input_ids=decoder_input_ids,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

# Décoder les tokens de sortie en texte lisible
result = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(result)
