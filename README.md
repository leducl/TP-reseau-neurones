lien du tp : https://sleek-think.ovh/enseignement/neural-network-perceptron/ 

## Objectifs du TP

- Comprendre le fonctionnement du perceptron simple  
- Implémenter l’algorithme du perceptron  
- Analyser les limites du perceptron sur des problèmes non-linéairement séparables  
- Appliquer le perceptron à des données réelles  

## Livrables attendus

### 1. Code

- Implémentation complète de la classe **PerceptronSimple**  
- Implémentation de la classe **PerceptronMultiClasse**  
- Scripts de test et de visualisation  

### 2. Rapport

- **Introduction** : Contexte et objectifs  
- **Méthodes** : Description des algorithmes implémentés  
- **Résultats** :  
  - Tests sur fonctions logiques  
  - Analyse de convergence  
  - Évaluation sur données réelles  
- **Discussion** :  
  - Limites du perceptron  
  - Cas d’usage appropriés  
- **Conclusion** : Synthèse des apprentissages  

### 3. Visualisations

- Graphiques de convergence  
- Visualisation des droites de séparation  
- Matrices de confusion  
- Comparaisons de performances  



Introduction

Le fichier README.md rappelle les objectifs pédagogiques : comprendre le fonctionnement du perceptron, l’implémenter et analyser ses limites, puis appliquer l’algorithme sur des données réelles.

Méthodes

Le perceptron simple initialise un vecteur de poids aléatoire puis itère sur les exemples pour ajuster poids et biais en fonction de l’erreur de prédiction.
La prédiction applique ensuite la règle de Heaviside pour renvoyer 0 ou 1.

Le perceptron multi‑classe entraîne un perceptron binaire par classe selon la stratégie « un‑contre‑tous », puis détermine la classe prédite en prenant le score maximal parmi ces modèles.

Des scripts annexes générent des données linéairement séparables et étudient la convergence pour différents taux d’apprentissage.

L’évaluation avancée combine un ensemble d’entraînement, de validation et de test avec un mécanisme d’early stopping basé sur partial_fit et la sauvegarde des meilleurs poids.

Résultats

PerceptronSimple.py comporte un jeu de tests sur les fonctions logiques AND, OR et XOR, affichant le score et le nombre d’époques nécessaires pour converger.

Le script de génération de données séparables affiche la performance finale du perceptron et la frontière de décision apprise.

Le perceptron multi‑classe dispose d’un test de base mesurant l’accuracy sur un jeu synthétique à trois classes.

Le fichier d’évaluation permet de calculer des matrices de confusion et un rapport de classification détaillé sur train/val/test, mais il requiert des méthodes (partial_fit, get_weights, set_weights) encore absentes.

Discussion

Les réponses théoriques notent que le perceptron échoue dès que les données ne sont pas linéairement séparables (exemple de XOR) et qu’il est sensible au bruit ou au déséquilibre des classes. Pour compenser, on peut envisager la normalisation des données ou l’ajout d’une couche cachée pour des problèmes non linéaires. On souligne également la dépendance aux taux d’apprentissage et à l’initialisation aléatoire.

À l’examen du code, plusieurs points restent perfectibles :

les labels sont parfois codés en -1/1 dans les jeux de données tandis que PerceptronSimple attend 0/1;

plusieurs scripts contiennent des lignes parasites de terminal à la fin, ce qui provoquerait une erreur d’exécution (exemple dans charger_donnees_iris_binaire.py) ;

l’évaluation multi‑classe ne peut aboutir tant que les méthodes manquantes ne sont pas implémentées.

Conclusion

Le dépôt fournit l’essentiel des éléments demandés : perceptron simple et multi‑classe, scripts de génération de données, analyse de convergence et réponses théoriques. Néanmoins, pour un fonctionnement complet, il faudrait uniformiser les conventions de labels, corriger les fins de fichier corrompues et ajouter les méthodes requises dans PerceptronMultiClasse. Une fois ces ajustements réalisés, l’ensemble devrait permettre de mener à bien l’évaluation sur les données réelles et de produire les visualisations prévues.



