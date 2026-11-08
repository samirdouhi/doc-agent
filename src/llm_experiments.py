
# Importer la fonction chat de la bibliothèque Ollama.
from ollama import chat

# Définir le nom du modèle de langage local.
model_name = "qwen2.5:7b"

# Définir les instructions générales du modèle.
# Le prompt système précise son rôle et son comportement.
system_prompt = (
    "Tu es un professeur d'informatique. "
    "Réponds en français avec des explications simples."
)

# Définir la question posée par l'utilisateur.
user_prompt = "Explique ce qu'est Docker en deux phrases."

# Construire la conversation envoyée au modèle.
# Chaque dictionnaire représente un message.
messages = [
    {
        # Instructions générales du modèle.
        "role": "system",
        "content": system_prompt
    },
    {
        # Question de l'utilisateur.
        "role": "user",
        "content": user_prompt
    }
]

# Tester deux valeurs différentes de température.
# La boucle exécute le même appel deux fois.
for temperature in [0.0, 1.0]:

    # Envoyer les messages à Ollama.
    # Le modèle génère une réponse selon la température.
    response = chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": temperature
        }
    )

    # Afficher la température utilisée.
    print(f"\nTempérature : {temperature}")

    # Afficher la réponse produite par le LLM.
    print(response.message.content)
