import openai

openai.api_key ='sk-proj-R7qE6cH1tMvOrD7yloWiT3BlbkFJBM3GAiU9UgwnV7aKCTKh'

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="tts-1",  # Vous pouvez aussi utiliser "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

# Exemple d'utilisation
prompt = "Comment faire une requête à ChatGPT depuis Python ?"
response = generate_response(prompt)
print(response)