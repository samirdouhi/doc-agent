# Importer la fonction chat de la bibliothèque ollama.
# Cette fonction permet à Python de communiquer avec Ollama.
from ollama import chat

# Définir le modèle de langage (LLM) que nous voulons utiliser.
# qwen2.5:7b est un modèle pré-entraîné d'environ 7 milliards de paramètres.
model_name = "qwen2.5:7b"

# Envoyer une question au modèle Qwen2.5 grâce à Ollama.
# La fonction chat() transmet les messages et récupère le résultat.
# Le résultat est stocké dans la variable response.
response = chat(

    # Indiquer à Ollama quel modèle utiliser.
    model=model_name,

    # Définir la liste des messages envoyés au modèle.
    messages=[
        {
            # Préciser que le message provient de l'utilisateur.
            "role": "user",

            # Contenu de la question envoyée au LLM.
            "content": "Explique en une phrase ce qu'est un réseau de neurones."
        }
    ]
)

# Récupérer le texte généré par Qwen2.5.
# response contient le résultat retourné par Ollama.
# message représente le message généré.
# content contient le texte de la réponse.
# print() affiche cette réponse dans le terminal.
print(response.message.content)