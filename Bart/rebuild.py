from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.models.bart.modeling_bart import BartScaledWordEmbedding, BartLearnedPositionalEmbedding, BartEncoderLayer
from tools import *
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = "facebook/bart-large"

tokenizer = BartTokenizer.from_pretrained(model)
model = BartForConditionalGeneration.from_pretrained(model, output_hidden_states=True)
model.to(device)

self = model.model.encoder

def embedding_step():
    self = model.model.encoder

    embed_dim = self.config.d_model
    embed_scale = math.sqrt(embed_dim) if self.config.scale_embedding else 1.0

    input_text = "He is a good man. You have no sympathy. You are my friend."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    encoder_outputs, embedding = get_embedding(inputs, model)

    attention_mask = inputs['attention_mask']
    attention_mask = attention_mask.to(dtype=torch.float)
    output_attentions = self.config.output_attentions

    input_ids = inputs['input_ids']
    input= input_ids
    #input_ids = input_ids.view(-1, input_ids[-1])

    self.layers = nn.ModuleList([BartEncoderLayer(self.config) for _ in range(self.config.encoder_layers)]).to(device)
        

    # max_length = 1024
    self.embed_positions = BartLearnedPositionalEmbedding(
            self.config.max_position_embeddings,
            embed_dim,
        )
    positional_embedding = self.embed_positions(input).to(device)
    print("positional embedding : \n" + str(positional_embedding))

    #add randomness but why ?
    
    self.embed_tokens = BartScaledWordEmbedding(
            self.config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
        ).to(device)
    self.embed_tokens = nn.Embedding(50265, embed_dim, self.padding_idx).to(device)
    
    inputs_embeddings = self.embed_tokens(input_ids).to(device)
    print("inputs embeddings : \n" + str(inputs_embeddings))

    hidden_states = inputs_embeddings + positional_embedding
    print("hidden states : \n" + str(hidden_states))

    self.layernorm_embedding = nn.LayerNorm(embed_dim).to(device)
    normalized_hidden_states = self.layernorm_embedding(hidden_states)
    print("normalized hidden states : \n" + str(normalized_hidden_states))

    encoder_states =  ()
    for encoder_layer in self.layers:
        final_hidden_states = encoder_layer(normalized_hidden_states, attention_mask, output_attentions=output_attentions, layer_head_mask=None)
        encoder_states =  encoder_states + (hidden_states,) 
    final_hidden_states = final_hidden_states[0]
    print("final hidden states : \n" + str(final_hidden_states))

    print("Original embedding : \n" + str(encoder_outputs.last_hidden_state))


    print("Original : \n" + str(batch_decode_embedding(encoder_outputs, model, tokenizer)))
    print(decode_embedding(encoder_outputs, model, tokenizer))
    encoder_states =  encoder_states + (hidden_states,) 
    encoder_outputs.last_hidden_state = final_hidden_states
    encoder_outputs.hidden_states = encoder_states
    print("Rebuild : \n" + str(batch_decode_embedding(encoder_outputs, model, tokenizer)))
    print(decode_embedding(encoder_outputs, model, tokenizer))


embedding_step()