
# ============================================================
# DOCAGENT - INTERFACE STREAMLIT (VERSION RAPIDE)
# ============================================================

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Utiliser notre agent rapide.
AGENT_MODULE = "src.agent_tool_calling_rapide"

st.set_page_config(
    page_title="DocAgent",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# 2. EXECUTER NOTRE AGENT
# ============================================================

def interroger_docagent(question: str):
    """Lancer DocAgent et récupérer sa réponse et ses sources."""

    # Utiliser Python de notre environnement .venv.
    commande = [
        sys.executable,
        "-m",
        AGENT_MODULE
    ]

    # Configurer l'encodage UTF-8.
    environnement = os.environ.copy()
    environnement["PYTHONIOENCODING"] = "utf-8"
    environnement["PYTHONUTF8"] = "1"

    # Exécuter l'agent rapide.
    resultat = subprocess.run(
        commande,
        input=question + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_DIR,
        env=environnement,
        timeout=240,
        check=False
    )

    # Récupérer la sortie du terminal.
    journal = resultat.stdout or ""

    if resultat.stderr:
        journal += "\n=== ERREURS ===\n" + resultat.stderr

    # ========================================================
    # EXTRAIRE LA REPONSE DE QWEN
    # ========================================================

    reponse = ""

    marqueur_reponse = "=== REPONSE DOCAGENT ==="

    if marqueur_reponse in journal:

        reponse = journal.split(
            marqueur_reponse,
            1
        )[1]

        reponse = reponse.split(
            "=== SOURCES OFFICIELLES ===",
            1
        )[0]

        reponse = reponse.replace(
            "Qwen génère sa réponse...",
            "",
            1
        ).strip()

    # ========================================================
    # RECUPERER LES SOURCES OFFICIELLES
    # ========================================================

    sources = []

    marqueur_sources = "=== PAGES CONSULTEES ==="

    if marqueur_sources in journal:

        partie_sources = journal.split(
            marqueur_sources,
            1
        )[1]

        for ligne in partie_sources.splitlines():

            url = ligne.strip()

            if url.startswith("https://"):

                if url not in sources:
                    sources.append(url)

    return reponse, sources, journal, resultat.returncode


# ============================================================
# 3. CREER L'INTERFACE STREAMLIT
# ============================================================

st.title("🤖 DocAgent")

st.write(
    "Assistant IA basé sur les documentations "
    "informatiques officielles."
)

st.caption(
    "Machine Learning · Qwen2.5:3b · Tool Calling"
)

st.divider()


# ============================================================
# 4. INITIALISER L'HISTORIQUE
# ============================================================

if "historique" not in st.session_state:
    st.session_state.historique = []


# ============================================================
# 5. BARRE LATERALE
# ============================================================

with st.sidebar:

    st.subheader("Paramètres")

    st.info(
        "Version rapide : agent_tool_calling_rapide.py"
    )

    if st.button("Effacer la conversation"):

        st.session_state.historique = []

        st.rerun()


# ============================================================
# 6. AFFICHER LES ANCIENS MESSAGES
# ============================================================

for echange in st.session_state.historique:

    # Message utilisateur.
    with st.chat_message("user"):
        st.write(echange["question"])

    # Réponse DocAgent.
    with st.chat_message("assistant"):

        st.markdown(echange["reponse"])

        # Afficher les sources.
        if echange["sources"]:

            st.markdown("**Sources officielles :**")

            for url in echange["sources"]:
                st.write(url)


# ============================================================
# 7. CHAMP DE QUESTION
# ============================================================

question = st.chat_input(
    "Pose ta question informatique..."
)


# ============================================================
# 8. TRAITER UNE NOUVELLE QUESTION
# ============================================================

if question:

    # Afficher la question.
    with st.chat_message("user"):
        st.write(question)

    # Afficher la réponse.
    with st.chat_message("assistant"):

        journal = ""
        sources = []
        reponse = ""

        try:

            # Indicateur pendant le traitement.
            with st.spinner(
                "DocAgent recherche et prépare sa réponse..."
            ):

                reponse, sources, journal, code = (
                    interroger_docagent(question)
                )

            # =================================================
            # VERIFIER LE RESULTAT
            # =================================================

            if code != 0:

                st.error(
                    "L'agent a rencontré une erreur. "
                    "Consulte le journal technique."
                )

                reponse = "Erreur d'exécution de DocAgent."

            elif reponse:

                # Afficher la réponse formatée.
                st.markdown(reponse)

            else:

                reponse = (
                    "DocAgent n'a pas pu produire "
                    "de réponse documentée."
                )

                st.warning(reponse)

            # =================================================
            # AFFICHER LES SOURCES
            # =================================================

            if sources:

                st.markdown("**Sources officielles :**")

                for url in sources:
                    st.write(url)

            # =================================================
            # JOURNAL TECHNIQUE
            # =================================================

            with st.expander("Voir le journal technique"):

                st.code(
                    journal,
                    language="text"
                )

        except subprocess.TimeoutExpired:

            reponse = (
                "DocAgent a dépassé la durée "
                "maximale de 4 minutes."
            )

            st.error(reponse)

        except Exception as erreur:

            reponse = (
                "Une erreur est survenue "
                "lors de l'exécution."
            )

            st.error(
                f"{reponse} Détail : {erreur}"
            )

        # ====================================================
        # SAUVEGARDER L'ECHANGE
        # ====================================================

        st.session_state.historique.append({
            "question": question,
            "reponse": reponse,
            "sources": sources
        })
