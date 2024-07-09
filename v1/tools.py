import torch
import random

def get_embedding(inputs, model):
    with torch.no_grad():
        encoder_outputs = model.model.encoder(**inputs)
        embedding = encoder_outputs.last_hidden_state
    return encoder_outputs, embedding[:,1:2, :].cpu().detach().numpy()[0][0]

def modif_embedding(embedding):
    embedding = modifier_aleatoirement(embedding)
    return embedding

def decode_embedding(encoder_outputs, model, tokenizer):
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
    
    return result

def modifier_aleatoirement(embedding):
    for i in range(len(embedding)):
        random_percent_change = random.uniform(-2, 2)
        # Calculer la nouvelle valeur modifiée
        new_value = embedding[i] * (1 + random_percent_change)
        embedding[i] = new_value
    return embedding