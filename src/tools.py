
# ============================================================
# DOCAGENT - OUTILS DE RECHERCHE ET DE SCRAPING
# ============================================================

# Importer DDGS pour rechercher des pages sur Internet.
from ddgs import DDGS

# Importer Requests pour télécharger les pages HTML.
import requests

# Importer BeautifulSoup pour analyser le HTML.
from bs4 import BeautifulSoup

# Importer re pour rechercher des mots dans les textes.
import re

# Importer urlsplit pour analyser les URL.
from urllib.parse import urlsplit

# Importer RobotFileParser pour lire les règles robots.txt.
from urllib.robotparser import RobotFileParser

# Importer notre liste blanche depuis config.py.
from src.config import OFFICIAL_SITES


# Nom utilisé par notre programme auprès des sites web.
USER_AGENT = "DocAgentLearningBot/0.1"


# ============================================================
# 1. VÉRIFICATION DES URL OFFICIELLES
# ============================================================

def is_official_url(url, domain):

    try:
        # Décomposer l'URL pour vérifier son vrai domaine.
        parsed = urlsplit(url)

        # Autoriser uniquement HTTPS et le domaine exact.
        return (
            parsed.scheme == "https"
            and parsed.hostname == domain
            and parsed.username is None
            and parsed.password is None
            and parsed.port in (None, 443)
        )

    except (ValueError, TypeError):
        # Refuser les URL invalides.
        return False


# ============================================================
# 2. RECHERCHE DES PAGES OFFICIELLES AVEC DDGS
# ============================================================

def search_docs(question, technology):

    # Vérifier que la technologie est autorisée.
    if technology not in OFFICIAL_SITES:
        print("Technologie non autorisée.")
        return []

    # Récupérer son domaine officiel.
    domain = OFFICIAL_SITES[technology]

    # Construire la requête de recherche.
    query = f"{question} site:{domain}"

    print("Recherche :", query)

    try:
        # Effectuer la recherche avec DDGS.
        results = DDGS(timeout=10).text(
            query,
            max_results=8
        )

    except Exception as error:
        # Gérer les erreurs du moteur de recherche.
        print("Erreur de recherche :", error)
        return []

    # Préparer une liste pour les pages autorisées.
    official_results = []

    # Parcourir les résultats trouvés.
    for result in results:

        # Récupérer le titre et l'URL.
        title = result.get("title", "")
        url = result.get("href", "")

        # Vérifier que le domaine est officiel.
        if is_official_url(url, domain):

            # Ajouter le résultat autorisé à la liste.
            official_results.append({
                "title": title,
                "url": url
            })

        # Conserver au maximum trois pages.
        if len(official_results) >= 3:
            break

    # Retourner la liste des pages officielles.
    return official_results


# ============================================================
# 3. TÉLÉCHARGEMENT ET EXTRACTION DES PAGES
# ============================================================

def fetch_page(url):

    # Vérifier que l'URL appartient à notre liste blanche.
    allowed = any(
        is_official_url(url, domain)
        for domain in OFFICIAL_SITES.values()
    )

    # Refuser les domaines non autorisés.
    if not allowed:
        print("URL non autorisée.")
        return None

    # Récupérer le domaine.
    domain = urlsplit(url).hostname

    # Construire l'adresse des règles robots.txt.
    robots_url = f"https://{domain}/robots.txt"

    # Définir le User-Agent du programme.
    headers = {
        "User-Agent": USER_AGENT
    }

    try:
        # Télécharger robots.txt pour vérifier les permissions.
        robots_response = requests.get(
            robots_url,
            headers=headers,
            timeout=10,
            allow_redirects=False
        )

        # Refuser si les règles ne peuvent pas être vérifiées.
        if robots_response.status_code != 200:
            print("Impossible de vérifier robots.txt.")
            return None

        # Analyser les règles du site.
        robots = RobotFileParser()
        robots.parse(robots_response.text.splitlines())

        # Vérifier que la consultation est autorisée.
        if not robots.can_fetch(USER_AGENT, url):
            print("Accès interdit par robots.txt.")
            return None

        # Télécharger la page web.
        response = requests.get(
            url,
            headers=headers,
            timeout=10,
            allow_redirects=False
        )

        # Refuser les redirections automatiques.
        if 300 <= response.status_code < 400:
            print("Redirection refusée.")
            return None

        # Vérifier les erreurs HTTP.
        response.raise_for_status()

        # Vérifier que le serveur a retourné du HTML.
        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type.lower():
            print("Le contenu n'est pas du HTML.")
            return None

        # Analyser le HTML avec BeautifulSoup.
        soup = BeautifulSoup(response.text, "html.parser")

        # Trouver le titre principal.
        title = soup.find("h1")

        title_text = (
            title.get_text(" ", strip=True)
            if title else ""
        )

        # Chercher d'abord article, sinon main.
        main = soup.find("article") or soup.find("main")

        if main is None:
            print("Contenu principal introuvable.")
            return None

        # Supprimer les éléments HTML secondaires.
        for element in main.select(
            "script, style, nav, footer, aside, "
            ".table-of-contents, .toc, "
            "[aria-label='Table of contents']"
        ):
            element.decompose()

        # Créer une liste contenant les blocs de texte.
        blocks = []

        # Récupérer les titres, paragraphes et blocs de code.
        for element in main.find_all(
            ["h2", "h3", "p", "pre"]
        ):

            # Préserver les lignes des blocs de code.
            separator = "\n" if element.name == "pre" else " "

            # Extraire le texte sans les balises HTML.
            text = element.get_text(separator, strip=True)

            # Éviter les blocs vides ou certains éléments de sommaire.
            if text and text.lower() not in {
                "table of contents",
                "skill level",
                "time to complete"
            }:
                blocks.append(text)

        # Si aucun bloc n'est trouvé, utiliser le texte général.
        if not blocks:
            blocks = [main.get_text(" ", strip=True)]

        # Regrouper les blocs pour obtenir le contenu.
        content = "\n\n".join(blocks)

        # Retourner la page sous forme de dictionnaire.
        return {
            "url": url,
            "title": title_text,
            "content": content[:30000],
            "blocks": blocks
        }

    except requests.exceptions.RequestException as error:
        # Gérer les problèmes de connexion et HTTP.
        print("Erreur de téléchargement :", error)
        return None


# ============================================================
# 4. EXTRAIRE LES PASSAGES PERTINENTS
# ============================================================

def extract_section(page, question, max_passages=3):

    # Vérifier que nous avons bien récupéré une page.
    if not page:
        print("Aucune page à analyser.")
        return []

    # Mots très fréquents qui aident peu à identifier un sujet.
    stop_words = {
        "how", "to", "the", "a", "an", "is", "are",
        "what", "with", "and", "for", "in", "of",
        "comment", "une", "un", "le", "la", "les",
        "de", "des", "du", "dans", "avec", "pour",
        "est", "que", "en", "sur", "et"
    }

    # Transformer la question en mots minuscules.
    question_words = set(
        re.findall(r"\w+", question.lower())
    )

    # Supprimer les mots trop courts et peu significatifs.
    question_words = {
        word for word in question_words
        if len(word) > 2 and word not in stop_words
    }

    # Préparer une liste pour les passages pertinents.
    matches = []

    # Parcourir les blocs extraits par fetch_page().
    for block in page["blocks"]:

        # Transformer chaque bloc en ensemble de mots.
        block_words = set(
            re.findall(r"\w+", block.lower())
        )

        # Compter les mots communs avec la question.
        score = len(question_words.intersection(block_words))

        # Conserver uniquement les blocs avec au moins un mot commun.
        if score > 0:

            # Enregistrer le score et le passage.
            matches.append({
                "score": score,
                "text": block,
                "url": page["url"]
            })

    # Classer les passages du plus pertinent au moins pertinent.
    matches.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Retourner au maximum trois passages.
    return matches[:max_passages]


# ============================================================
# 5. TEST DES TROIS FONCTIONS
# ============================================================

if __name__ == "__main__":

    # Question utilisée pour notre test.
    question = "How to build a Docker image using Dockerfile"

    # Technologie choisie manuellement.
    technology = "docker"

    # --------------------------------------
    # TEST 1 : SEARCH_DOCS
    # --------------------------------------

    print("\n=== TEST 1 : RECHERCHE ===")

    pages = search_docs(question, technology)

    for page in pages:
        print("\nTitre :", page["title"])
        print("URL :", page["url"])

    # --------------------------------------
    # TEST 2 : FETCH_PAGE
    # --------------------------------------

    print("\n=== TEST 2 : SCRAPING ===")

    # Page connue de la documentation Docker.
    url = (
        "https://docs.docker.com/get-started/"
        "docker-concepts/building-images/"
    )

    # Télécharger et analyser la page.
    document = fetch_page(url)

    if document:

        # Afficher les informations récupérées.
        print("\nTitre :", document["title"])
        print("Nombre de blocs :", len(document["blocks"]))

        # --------------------------------------
        # TEST 3 : EXTRACT_SECTION
        # --------------------------------------

        print("\n=== TEST 3 : EXTRACTION PERTINENTE ===")

        # Sélectionner les passages liés à notre question.
        passages = extract_section(
            document,
            question,
            max_passages=3
        )

        # Afficher les passages sélectionnés.
        for index, passage in enumerate(passages, start=1):

            print(f"\nPassage {index}")
            print("Score :", passage["score"])
            print("Texte :", passage["text"][:600])
            print("Source :", passage["url"])

        # Aucun passage correspondant à la question.
        if not passages:
            print("Aucun passage pertinent trouvé.")

    else:
        print("Impossible de récupérer la documentation.")
