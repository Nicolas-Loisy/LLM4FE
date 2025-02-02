# TODO

## Étape 1 : Génération de données avec un LLM
- Fournir un dataset initial au LLM
- Demander au LLM de générer un dataset complet en reprenant les données avec ou sans transformation
- Vérifier la cohérence et la qualité des données générées

## Étape 2 : Structured Output
- Demander au LLM de fournir un format de structured output (JSON, CSV, etc.) en lui donnant les colonnes et quelques lignes d'exemple
- Fournir les données au fur et à mesure pour observer l'adaptation du modèle
- Comparer l'efficacité du structured output avec l'approche initiale
- Ajuster le prompt et les paramètres pour optimiser les résultats


## Notes : 
Ne pas chercher à transformer les données directement, mais fournir un dataset initial et obtenir en retour un dataset entièrement généré. 
Au passage, le modèle pourra modifier les données en les reprenant avec ou sans transformation. 
On peut également lui demander de proposer un format de structured output (en lui fournissant les colonnes et quelques lignes d’exemple), puis lui transmettre progressivement les données.
