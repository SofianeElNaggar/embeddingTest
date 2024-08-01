from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.models.bart.modeling_bart import BartScaledWordEmbedding, BartLearnedPositionalEmbedding, BartEncoderLayer
from tools import *
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = "facebook/bart-large"

#init model
tokenizer = BartTokenizer.from_pretrained(model)
model = BartForConditionalGeneration.from_pretrained(model, output_hidden_states=True)
model.to(device)

#init input
input_text = "He is a good man. You have no sympathy. You are my friend."
inputs = tokenizer(input_text, return_tensors="pt").to(device)
encoder_outputs, embedding = get_embedding(inputs, model)

#variable config
self = model.model.encoder
config = self.config

#init encoder
self.dropout = config.dropout
embed_dim = config.d_model
self.padding_idx = config.pad_token_id
self.max_source_positions = config.max_position_embeddings
embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

self.embed_tokens = BartScaledWordEmbedding(
    config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
).to(device)

self.embed_positions = BartLearnedPositionalEmbedding(
    config.max_position_embeddings,
    embed_dim,
).to(device)

self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)]).to(device)
self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
self._use_sdpa = config._attn_implementation == "sdpa"
self.layernorm_embedding = nn.LayerNorm(embed_dim).to(device)

self.gradient_checkpointing = False
self.post_init()

output_attentions =  self.config.output_attentions
output_hidden_states = (self.config.output_hidden_states)
return_dict = self.config.use_return_dict

input_ids = inputs['input_ids']
input = input_ids
input_ids = input_ids.view(-1, input_ids.shape[-1])

inputs_embeds = self.embed_tokens(input_ids).to(device)
print("inputs embeddings : \n" + str(inputs_embeds))

embed_pos = self.embed_positions(input).to(device)
print("positional embedding : \n" + str(embed_pos))

hidden_states = inputs_embeds + embed_pos
hidden_states = self.layernorm_embedding(hidden_states).to(device)
print("normalized hidden states : \n" + str(hidden_states))

encoder_states = ()
all_attentions = ()

attention_mask = inputs['attention_mask']
attention_mask = attention_mask.to(dtype=torch.float)

for idx, encoder_layer in enumerate(self.layers):
    encoder_states = encoder_states + (hidden_states,)
    layer_outputs = encoder_layer(
        hidden_states,
        attention_mask,
        layer_head_mask=None,
        output_attentions=output_attentions,
    )
    
    hidden_states = layer_outputs[0]
    if output_attentions:
        all_attentions = all_attentions + (layer_outputs[1],)
    encoder_states =  encoder_states + (hidden_states,) 
    
print("final hidden states : \n" + str(hidden_states))
    
print("Original embedding : \n" + str(encoder_outputs.last_hidden_state))
print("Original : \n" + str(batch_decode_embedding(encoder_outputs, model, tokenizer)))
print(decode_embedding(encoder_outputs, model, tokenizer))
encoder_outputs.last_hidden_state = hidden_states
encoder_outputs.hidden_states = encoder_states
encoder_outputs.attentions = all_attentions
print("Rebuild : \n" + str(batch_decode_embedding(encoder_outputs, model, tokenizer)))
print(decode_embedding(encoder_outputs, model, tokenizer))
    