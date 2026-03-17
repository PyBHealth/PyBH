Bienvenue sur la documentation PyBH
===================================

Présentation du projet
----------------------
PyBH est une bibliothèque open-source de modélisation statistique bayésienne, conçue pour le secteur de la santé. 
Ce projet cherche à développer une preuve de concept pour démocratiser l'analyse de survie bayésienne.

* **Le constat** : Les outils actuels (fréquentistes) fournissent des estimations figées, masquant souvent l'incertitude réelle des données cliniques.
* **Notre solution** : Une librairie Python basée sur PyMC qui simplifie l'usage des statistiques bayésiennes.
* **L'objectif** : Offrir aux praticiens une interface intuitive style Scikit-learn pour une évaluation plus fine des risques et une aide à la décision médicale sécurisée par une meilleure gestion de l'incertitude.

Statistiques Fréquentistes vs Bayésiennes
-----------------------------------------

Dans PyBH, nous offrons une preuve de concept permettant aux utilisateurs de choisir celle qui correspond le mieux à leurs besoins et à leur compréhension des données:

1. **L'approche Fréquentiste (via Lifelines)**
    * **Définition** : La probabilité est vue comme la fréquence limite d'un événement.
    * **Paramètres** : Ils sont considérés comme des valeurs fixes mais inconnues.
    * **Objectif** : Chercher l'estimation ponctuelle la plus probable (Maximum de Vraisemblance).
    * **Incertitude** : Exprimée par des intervalles de confiance classiques.

2. **L'approche Bayésienne (via PyMC)**
    * **Définition** : La probabilité est vue comme un degré de croyance ou une mesure de l'incertitude.
    * **Paramètres** : Ce sont des variables aléatoires qui suivent une distribution de probabilité.
    * **Le Prior** : Permet d'intégrer des connaissances expertes ou historiques avant l'analyse.
    * **Le Posterior** : Fournit une image complète de l'incertitude via une distribution de probabilité.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Table des Matières:

   Survival_Analysis/index

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Metrics:
   :hidden:

   Metrics/Metrics