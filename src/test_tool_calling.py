
# ============================================================
# DOCAGENT - TEST DU TOOL CALLING AVEC QWEN2.5
# ============================================================

# Manipuler les données JSON.
import json

# Gérer les chemins de fichiers.
from pathlib import Path

# Charger notre modèle Machine Learning.
from joblib import load

# Communiquer avec Qwen2.5 via Ollama.
from ollama import chat

# Importer notre liste blanche.
from src.config import OFFICIAL_SITES

# Importer notre outil de recherche.
from src.tools import search_docs


# ============================================================
# 1. CHARGER LE MODELE MACHINE LEARNING
# ============================================================

# Récupérer le dossier principal du projet.
project_dir = Path(__file__).resolve().parents[1]

# Chemin du modèle déjà entraîné.
model_path = project_dir / "models" / "docagent_router.joblib"

# Vérifier que le modèle existe.
if not model_path.exists():
    print("Modèle ML introuvable :", model_path)
    raise SystemExit()

# Charger uniquement notre modèle local de confiance.
model = load(model_path)


# ============================================================
# 2. RECUPERER LA QUESTION DE L'UTILISATEUR
# ============================================================

question = input(
    "\nPose une question informatique : "
).strip()

# Vérifier que la question n'est pas vide.
if not question:
    print("Question vide.")
    raise SystemExit()


# ============================================================
# 3. PREDIRE LA TECHNOLOGIE AVEC NOTRE MODELE ML
# ============================================================

technology = str(model.predict([question])[0])

print("\n=== PREDICTION ML ===")
print("Question :", question)
print("Technologie prédite :", technology)

# Vérifier que cette technologie est prise en charge.
if technology not in OFFICIAL_SITES:
    print("Technologie non autorisée.")
    raise SystemExit()

print("Site officiel :", OFFICIAL_SITES[technology])


# ============================================================
# 4. DEFINIR L'OUTIL ACCESSIBLE A QWEN
# ============================================================

def chercher_pages(question_recherche: str) -> str:
    """
    Rechercher des pages de documentation officielle.

    Args:
        question_recherche: Question informatique à rechercher.

    Returns:
        Une liste JSON de titres et d'URL officielles.
    """

    # Première recherche avec la question fournie.
    pages = search_docs(
        question_recherche,
        technology
    )

    # ========================================================
    # RECHERCHE ALTERNATIVE POUR LES BRANCHES GIT
    # ========================================================

    # Mettre la question en minuscules.
    mots = question_recherche.lower()

    # Si Git est prédit, qu'aucune page n'a été trouvée
    # et que la question concerne les branches :
    if (
        not pages
        and technology == "git"
        and ("branche" in mots or "branch" in mots)
    ):

        print("\nAucun résultat avec la première recherche.")
        print("Nouvelle recherche en anglais...")

        # Réessayer avec une requête plus standard.
        pages = search_docs(
            "git create branch",
            technology
        )

    # Renvoyer les résultats au format JSON.
    return json.dumps(
        pages,
        ensure_ascii=False
    )


# ============================================================
# 5. PREPARER LES MESSAGES POUR QWEN
# ============================================================

messages = [
    {
        "role": "system",
        "content": (
            "Tu es DocAgent, un assistant de documentation "
            "informatique officielle. "
            "Pour rechercher la documentation, demande "
            "l'outil chercher_pages. "
            "Cet outil te retournera des titres et des URL. "
            "N'invente jamais de sources. "
            "Ne donne pas encore de procédure technique, "
            "car le contenu des pages n'a pas été lu."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 6. DEMANDER A QWEN DE CHOISIR UN OUTIL
# ============================================================

print("\n=== QWEN DECIDE ===")

try:

    response = chat(
        model="qwen2.5:7b",
        messages=messages,
        tools=[chercher_pages],
        options={
            "temperature": 0
        }
    )

except Exception as error:

    print("Erreur de communication avec Ollama :", error)
    raise SystemExit()


# ============================================================
# 7. VERIFIER SI QWEN A DEMANDE UN OUTIL
# ============================================================

# Conserver le message de Qwen.
messages.append(response.message)

if not response.message.tool_calls:

    print("Qwen n'a demandé aucun outil.")
    print("Aucune recherche n'a été exécutée.")
    raise SystemExit()


# ============================================================
# 8. RECUPERER L'APPEL DE L'OUTIL
# ============================================================

# Pour ce premier test, nous traitons un seul appel.
tool_call = response.message.tool_calls[0]

# Récupérer le nom de l'outil demandé.
tool_name = tool_call.function.name

print("Outil demandé par Qwen :", tool_name)

# Refuser tout outil non autorisé.
if tool_name != "chercher_pages":

    print("Outil non autorisé :", tool_name)
    raise SystemExit()


# ============================================================
# 9. EXECUTER L'OUTIL AVEC PYTHON
# ============================================================

print("\n=== EXECUTION DE L'OUTIL ===")

# Utiliser la question originale pour ce test.
# Le domaine officiel reste fixé par notre routeur ML.
result = chercher_pages(question)

# Transformer le JSON en liste Python.
pages_trouvees = json.loads(result)


# ============================================================
# 10. VERIFIER SI DES SOURCES ONT ETE TROUVEES
# ============================================================

if not pages_trouvees:

    print("\n=== DOCUMENTATION INTROUVABLE ===")

    print("Aucune source officielle récupérée.")

    print(
        "DocAgent ne peut pas fournir "
        "de réponse technique vérifiée."
    )

    # Arrêter le programme.
    # Ainsi, Qwen ne répond pas à partir de sa mémoire.
    raise SystemExit()


# ============================================================
# 11. AFFICHER LES RESULTATS DE L'OUTIL
# ============================================================

print("\n=== RESULTAT DE L'OUTIL ===")

for index, page in enumerate(
    pages_trouvees,
    start=1
):

    print(f"\nPage {index}")
    print("Titre :", page["title"])
    print("URL :", page["url"])


# ============================================================
# 12. TRANSMETTRE LES RESULTATS A QWEN
# ============================================================

# Ajouter le résultat réel de l'outil à la conversation.
messages.append({
    "role": "tool",
    "tool_name": tool_name,
    "content": result
})

print("\n=== QWEN ANALYSE LES RESULTATS ===")

try:

    final_response = chat(
        model="qwen2.5:7b",
        messages=messages,
        options={
            "temperature": 0
        }
    )

except Exception as error:

    print("Erreur pendant la génération :", error)
    raise SystemExit()


# ============================================================
# 13. AFFICHER LES SOURCES VERIFIEES
# ============================================================

# Pour ce test, nous n'affichons pas la réponse technique
# de Qwen. En effet, nous n'avons récupéré que des URL,
# pas encore le contenu réel des documentations.

print("\n=== SOURCES OFFICIELLES DISPONIBLES ===")

for index, page in enumerate(
    pages_trouvees,
    start=1
):

    print(f"[{index}] {page['title']}")
    print(page["url"])

print("\n=== TOOL CALLING TERMINE ===")

print("Qwen a demandé un outil.")
print("Python a exécuté cet outil.")
print("Les résultats ont été transmis à Qwen.")

print(
    "\nProchaine étape : ajouter la lecture "
    "des pages et l'extraction des passages."
)
