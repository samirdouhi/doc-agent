
# Importer Requests pour télécharger une page web.
import requests

# Importer BeautifulSoup pour analyser le HTML.
from bs4 import BeautifulSoup

# Définir l'adresse de la documentation officielle Docker.
url = "https://docs.docker.com/get-started/docker-concepts/building-images/"

# Télécharger le contenu HTML de la page.
response = requests.get(url, timeout=10)

# Vérifier qu'aucune erreur HTTP n'est survenue.
response.raise_for_status()

# Créer l'objet soup à partir du HTML téléchargé.
# Cet objet permet de rechercher des éléments HTML.
soup = BeautifulSoup(response.text, "html.parser")

# Rechercher le titre principal h1.
title = soup.find("h1")

# Afficher le titre s'il existe.
if title:
    print("Titre :", title.get_text(strip=True))

# Rechercher la partie principale de la page.
main_content = soup.find("main")

# Vérifier si cette partie existe.
if main_content:

    # Récupérer les trois premiers paragraphes.
    paragraphs = main_content.find_all("p", limit=3)

    # Parcourir les paragraphes un par un.
    for paragraph in paragraphs:

        # Extraire le texte sans les balises HTML.
        text = paragraph.get_text(" ", strip=True)

        # Afficher le texte s'il n'est pas vide.
        if text:
            print("\nParagraphe :", text)

else:
    print("Contenu principal introuvable.")
