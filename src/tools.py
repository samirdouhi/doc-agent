
# ============================================================
# DOCAGENT - OUTILS DE RECHERCHE ET DE SCRAPING
# ============================================================

# Rechercher des pages avec DDGS.
from ddgs import DDGS

# Télécharger le contenu des pages web.
import requests

# Analyser le code HTML.
from bs4 import BeautifulSoup

# Rechercher des mots dans le texte.
import re

# Analyser les adresses URL.
from urllib.parse import urlsplit

# Analyser les règles robots.txt.
from urllib.robotparser import RobotFileParser

# Importer les domaines officiels autorisés.
from src.config import OFFICIAL_SITES


# Nom de notre programme envoyé aux sites web.
USER_AGENT = "DocAgentLearningBot/0.1"


# ============================================================
# 1. VERIFIER LES URL OFFICIELLES
# ============================================================

def is_official_url(url, domain):

    try:
        # Décomposer l'adresse URL.
        parsed = urlsplit(url)

        # Vérifier le protocole et le domaine exact.
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
# 2. RECHERCHER LES DOCUMENTATIONS OFFICIELLES
# ============================================================

def search_docs(question, technology):

    # Vérifier que la technologie est autorisée.
    if technology not in OFFICIAL_SITES:
        print("Technologie non autorisée.")
        return []

    # Récupérer le domaine officiel correspondant.
    domain = OFFICIAL_SITES[technology]

    # Construire la requête de recherche.
    query = f"{question} site:{domain}"

    print("Recherche :", query)

    try:
        # Rechercher jusqu'à 8 résultats.
        results = DDGS(timeout=10).text(
            query,
            max_results=8
        )

    except Exception as error:
        # Gérer les erreurs de recherche.
        print("Erreur de recherche :", error)
        return []

    # Liste des résultats officiels.
    official_results = []

    # Parcourir les résultats.
    for result in results:

        title = result.get("title", "")
        url = result.get("href", "")

        # Vérifier que chaque URL est autorisée.
        if is_official_url(url, domain):

            # Ajouter le résultat à notre liste.
            official_results.append({
                "title": title,
                "url": url
            })

        # Limiter à trois pages officielles.
        if len(official_results) >= 3:
            break

    return official_results


# ============================================================
# 3. TELECHARGER ET EXTRAIRE UNE PAGE
# ============================================================

def fetch_page(url):

    # Vérifier que l'URL appartient à notre liste blanche.
    allowed = any(
        is_official_url(url, domain)
        for domain in OFFICIAL_SITES.values()
    )

    if not allowed:
        print("URL non autorisée.")
        return None

    # Récupérer le domaine de la page.
    domain = urlsplit(url).hostname

    # Construire l'URL du fichier robots.txt.
    robots_url = f"https://{domain}/robots.txt"

    # Identifier notre programme.
    headers = {
        "User-Agent": USER_AGENT
    }

    try:

        # ====================================================
        # ETAPE A : VERIFIER ROBOTS.TXT
        # ====================================================

        robots_response = requests.get(
            robots_url,
            headers=headers,
            timeout=10,
            allow_redirects=False
        )

        # Récupérer le statut HTTP.
        status = robots_response.status_code

        print("Statut robots.txt :", status)

        # CAS 1 : robots.txt existe.
        if status == 200:

            # Créer l'analyseur de règles.
            robots = RobotFileParser()

            # Lire les règles du site.
            robots.parse(
                robots_response.text.splitlines()
            )

            # Vérifier que la page est autorisée.
            if not robots.can_fetch(USER_AGENT, url):
                print("Accès interdit par robots.txt.")
                return None

        # CAS 2 : robots.txt est absent.
        elif status in (404, 410):

            # L'absence de robots.txt ne constitue
            # pas une interdiction d'accès.
            print("robots.txt absent : accès possible.")

        # CAS 3 : autre statut.
        else:

            # Refuser les cas non pris en charge
            # pour éviter un accès non vérifié.
            print(
                "Impossible de vérifier robots.txt.",
                "Statut :", status
            )
            return None

        # ====================================================
        # ETAPE B : TELECHARGER LE HTML
        # ====================================================

        response = requests.get(
            url,
            headers=headers,
            timeout=10,
            allow_redirects=False
        )

        # Refuser les redirections non vérifiées.
        if 300 <= response.status_code < 400:
            print("Redirection refusée.")
            return None

        # Détecter les erreurs HTTP.
        response.raise_for_status()

        # Vérifier que le contenu est du HTML.
        content_type = response.headers.get(
            "Content-Type", ""
        )

        if "text/html" not in content_type.lower():
            print("Le contenu n'est pas du HTML.")
            return None

        # ====================================================
        # ETAPE C : ANALYSER LE HTML
        # ====================================================

        # Créer un objet BeautifulSoup.
        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Extraire le titre principal.
        title = soup.find("h1")

        title_text = (
            title.get_text(" ", strip=True)
            if title else ""
        )

        # ====================================================
        # ETAPE D : TROUVER LE CONTENU PRINCIPAL
        # ====================================================

        # Les documentations utilisent des structures
        # HTML différentes selon leur générateur.

        # Rechercher plusieurs conteneurs possibles.
        main = (
            soup.select_one("article")
            or soup.select_one("main")
            or soup.select_one("div.body[role='main']")
            or soup.select_one("div[role='main']")
            or soup.select_one("div.body")
            or soup.select_one("#content")
        )

        # Si aucun conteneur n'a été trouvé.
        if main is None:
            print("Contenu principal introuvable.")
            return None

        # ====================================================
        # ETAPE E : NETTOYER LE HTML
        # ====================================================

        # Supprimer les scripts, menus, pieds de page
        # et certains sommaires.
        for element in main.select(
            "script, style, nav, footer, aside, "
            ".table-of-contents, .toc, "
            "[aria-label='Table of contents']"
        ):
            element.decompose()

        # ====================================================
        # ETAPE F : EXTRAIRE LES BLOCS DE TEXTE
        # ====================================================

        blocks = []

        # Récupérer les titres, paragraphes et blocs de code.
        for element in main.find_all(
            ["h2", "h3", "p", "pre"]
        ):

            # Préserver les retours à la ligne du code.
            separator = (
                "\n" if element.name == "pre" else " "
            )

            # Extraire le texte du bloc.
            text = element.get_text(
                separator,
                strip=True
            )

            # Ignorer les blocs vides et certains titres inutiles.
            if text and text.lower() not in {
                "table of contents",
                "skill level",
                "time to complete"
            }:

                blocks.append(text)

        # Si aucun bloc n'est détecté, récupérer
        # le texte général du contenu principal.
        if not blocks:
            blocks = [
                main.get_text(" ", strip=True)
            ]

        # Regrouper les blocs en un seul texte.
        content = "\n\n".join(blocks)

        # ====================================================
        # ETAPE G : RETOURNER LE DOCUMENT
        # ====================================================

        return {
            "url": url,
            "title": title_text,
            "content": content[:30000],
            "blocks": blocks
        }

    except requests.exceptions.RequestException as error:

        # Gérer les erreurs réseau et HTTP.
        print("Erreur de téléchargement :", error)
        return None


# ============================================================
# 4. SELECTIONNER LES PASSAGES PERTINENTS
# ============================================================

def extract_section(page, question, max_passages=3):

    # Vérifier que le document existe.
    if not page:
        print("Aucune page à analyser.")
        return []

    # Mots fréquents à ignorer lors du classement.
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

    # Supprimer les mots fréquents ou trop courts.
    question_words = {
        word for word in question_words
        if len(word) > 2 and word not in stop_words
    }

    # Liste des passages sélectionnés.
    matches = []

    # Parcourir les blocs de la documentation.
    for block in page["blocks"]:

        # Transformer le bloc en ensemble de mots.
        block_words = set(
            re.findall(r"\w+", block.lower())
        )

        # Compter les mots communs avec la question.
        score = len(
            question_words.intersection(block_words)
        )

        # Conserver les blocs ayant un score positif.
        if score > 0:

            matches.append({
                "score": score,
                "text": block,
                "url": page["url"]
            })

    # Classer les passages par pertinence lexicale.
    matches.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Retourner les meilleurs passages.
    return matches[:max_passages]


# ============================================================
# 5. TESTER NOS TROIS FONCTIONS
# ============================================================

if __name__ == "__main__":

    # Question de test.
    question = "How to create a branch"

    # Technologie choisie pour ce test isolé.
    technology = "git"

    # --------------------------------------------------------
    # TEST 1 : SEARCH_DOCS
    # --------------------------------------------------------

    print("\n=== TEST 1 : RECHERCHE ===")

    pages = search_docs(
        question,
        technology
    )

    # Afficher les résultats.
    for page in pages:

        print("\nTitre :", page["title"])
        print("URL :", page["url"])

    if not pages:
        print("Aucune page officielle trouvée.")

    # --------------------------------------------------------
    # TEST 2 : FETCH_PAGE
    # --------------------------------------------------------

    print("\n=== TEST 2 : SCRAPING ===")

    # Page officielle Git.
    url = "https://git-scm.com/docs/git-branch"

    # Télécharger la documentation.
    document = fetch_page(url)

    if document:

        print("\nTitre :", document["title"])
        print("Nombre de blocs :", len(document["blocks"]))

        # ----------------------------------------------------
        # TEST 3 : EXTRACT_SECTION
        # ----------------------------------------------------

        print("\n=== TEST 3 : EXTRACTION PERTINENTE ===")

        passages = extract_section(
            document,
            question,
            max_passages=3
        )

        # Afficher les passages.
        for index, passage in enumerate(
            passages,
            start=1
        ):

            print(f"\nPassage {index}")
            print("Score :", passage["score"])
            print("Texte :", passage["text"][:600])
            print("Source :", passage["url"])

        if not passages:
            print("Aucun passage pertinent trouvé.")

    else:
        print("Impossible de récupérer la documentation.")
