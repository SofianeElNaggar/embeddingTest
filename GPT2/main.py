from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

model_name = "openai-community/gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)


prompt = "king"

input_ids = tokenizer(prompt, return_tensors="pt").to(device)

print(model.config)

with torch.no_grad():
    outputs = model(**input_ids)


embedding = outputs.hidden_states[-1]
    
gen_tokens = model.generate(
    input_ids.input_ids,
    attention_mask = input_ids['attention_mask'],
    pad_token_id = 50256,
    do_sample=True,
    temperature=0.9,
    max_new_tokens=1
)

print(gen_tokens)

new_tensor = gen_tokens.narrow(1, 0, gen_tokens.size(1) - 1)


gen_text = tokenizer.decode(input_ids['input_ids'][0], skip_special_tokens=True)
print("text : " + str(gen_text))