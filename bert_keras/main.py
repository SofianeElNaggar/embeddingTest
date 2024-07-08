from transformers import TFBertModel, BertTokenizer
import tensorflow as tf

# Charger le tokenizer et le modèle BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Exemple de texte
text = "This is an example sentence for BERT embeddings."

# Tokeniser le texte
inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True)

# Passer les inputs au modèle
outputs = model(inputs)

# Le tenseur contenant les embeddings pour chaque token
last_hidden_states = outputs.last_hidden_state

# Convertir les embeddings en numpy array
embeddings = last_hidden_states.numpy()

# Afficher les embeddings pour chaque token
print(embeddings)

