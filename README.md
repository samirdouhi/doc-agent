## DocAgent — Assistant IA de recherche documentaire

**DocAgent** est un assistant IA que j’ai conçu et développé pour faciliter la recherche d’informations dans les documentations techniques officielles et fournir des réponses accompagnées de leurs sources.

### Mon rôle
J’ai assuré le développement du projet de bout en bout : préparation et nettoyage des données, entraînement et évaluation du modèle Machine Learning, conception du système de recherche documentaire, intégration d’un LLM local avec Tool Calling, optimisation des performances et développement de l’interface web.

### Composants clés
- **Machine Learning :** classification automatique des questions par technologie (Python, Git, Docker, JavaScript, .NET et SQL) avec TF-IDF et régression logistique.
- **LLM et Tool Calling :** intégration de Qwen2.5 avec Ollama pour orchestrer les outils Python et générer des réponses.
- **Recherche documentaire :** utilisation de DDGS, Requests et BeautifulSoup pour rechercher, consulter et extraire des informations depuis les documentations officielles.
- **Interface web :** développement d’une application interactive avec Streamlit.
- **Technologies :** Python, Scikit-learn, Pandas, Ollama, Qwen2.5, Streamlit et Git.

### Résultats obtenus
Le modèle ML a atteint un **F1-score macro de 80,5 %** sur un jeu de test de 1 213 questions issues du dataset préparé à partir de Stack Overflow.

L’optimisation de l’agent a permis de réduire le temps de réponse observé d’environ **120 secondes à 15 secondes** sur les tests réalisés, grâce à l’utilisation d’un modèle plus léger et à la réduction du nombre d’appels au LLM.

Le projet dispose actuellement d’une **première version fonctionnelle (V1)**, avec des améliorations prévues concernant la compréhension multilingue, la pertinence des recherches et la vérification des réponses générées.
