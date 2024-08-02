from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def decode(model, tokenizer, input_ids, inputs_embeds):
    outputs = model.generate(input_ids, max_new_tokens=len(inputs_embeds[0]))
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    batch_text = tokenizer.batch_decode(outputs[0])
    return text, batch_text

def set_new_embedding(model, device, input_embeddings, input_ids, modified_embeds):
    # Réimplanter les embeddings modifiés dans le modèle
    # Nous devons nous assurer que nous réimplantons une couche d'embedding de la même taille que l'original
    vocab_size, embedding_dim = input_embeddings.weight.shape
    new_embeddings = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
    new_embeddings.weight.data.copy_(input_embeddings.weight.data)
    # Remplacer les embeddings pour les tokens spécifiques (ici ceux de notre texte)
    new_embeddings.weight.data[input_ids] = modified_embeds.squeeze(0).to(device)

    model.set_input_embeddings(new_embeddings)