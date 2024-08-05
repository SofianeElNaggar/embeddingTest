from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import tools as tools
from collections import Counter
import numpy as np

def decode(model, tokenizer, input_ids, inputs_embeds):
    outputs = model.generate(input_ids, max_new_tokens=len(inputs_embeds[0]))
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    batch_text = tokenizer.batch_decode(outputs[0])
    return text, batch_text

def set_new_embedding(model, device, input_embeddings, input_ids, modified_embeds):
    vocab_size, embedding_dim = input_embeddings.weight.shape
    new_embeddings = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
    new_embeddings.weight.data.copy_(input_embeddings.weight.data)
    new_embeddings.weight.data[input_ids] = modified_embeds.squeeze(0).to(device)

    model.set_input_embeddings(new_embeddings)
    
def rotate_around_point(vector, distance):
    return tools.rotate_around_point_lin(vector, distance)

def find_neighbor_around(model, tokenizer, device, input_embeddings, inputs_embeds, input_ids, step=0.5, start_distance=0, min_lap=0, max_lap=1):
    n = 0
    
    embedding = inputs_embeds.cpu().detach().numpy()
    embedding_sep_token_1 = embedding[0][0]
    embedding_sep_token_2 = embedding[0][-1]
    embedding_no_sep_token = embedding[:,1:-1,:][0]
    
    if len(embedding_no_sep_token) != 1:
        return "this is not a word or this is not in your Bart vocabulary"
    
    if start_distance == 0:
        start_distance = step
    
    if min_lap > max_lap:
        max_lap = min_lap
        
    words = {}
    distance = start_distance
    while n <= min_lap or max_lap > n:
        
        list_embeddings = []
        
        embeddings = rotate_around_point(embedding_no_sep_token[0], distance)
        list_embeddings.append(embeddings)
        
        results = []
        
        for modified_embedding in list_embeddings[0]:
            modif = [[embedding_sep_token_1]]
            modif[0].append(modified_embedding)
            modif[0].append(embedding_sep_token_2)
            modif = np.array(modif)
            modified_embed = torch.FloatTensor(modif).to(device)
            #print(modified_embed.shape)
            set_new_embedding(model, device, input_embeddings, input_ids, modified_embed)
            word, batch = decode(model, tokenizer, input_ids, inputs_embeds)
            results.append(word)
            
        words["Distance : " + str(distance)] = dict(Counter(results))
        distance += step
        n += 1

    return words