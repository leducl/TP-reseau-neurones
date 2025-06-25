lien du tp : https://sleek-think.ovh/enseignement/neural-network-perceptron/ 

# TP Perceptron

## Table des matières
1. [Objectifs du TP](#objectifs-du-tp)  
2. [Livrables attendus](#livrables-attendus)  
   - [Code](#code)  
   - [Rapport](#rapport)  
   - [Visualisations](#visualisations)  
3. [Instructions de mise en route](#instructions-de-mise-en-route)  
4. [Licence](#licence)  

---

## Objectifs du TP

- Comprendre le fonctionnement du **perceptron simple**  
- Implémenter l’**algorithme du perceptron**  
- Analyser les **limites** du perceptron sur des problèmes **non-linéairement séparables**  
- Appliquer le perceptron à des **données réelles**  

---

## Livrables attendus

### 1. Code

- **`PerceptronSimple`**  
  - Implémentation complète de la classe  
  - Méthodes d’entraînement (`fit`), de prédiction (`predict`), et d’évaluation  
- **`PerceptronMultiClasse`**  
  - Implémentation du perceptron pour classification multiclasse (stratégie un-contre-tous ou un-contre-un)  
- **Scripts de test et de visualisation**  
  - Exemples d’application sur données synthétiques (fonctions logiques)  
  - Tests sur jeux de données réelles (ex. Iris, MNIST simplifié)  
  - Génération de graphiques de convergence et de frontières de décision  

### 2. Rapport

1. **Introduction**  
   - Contexte du TP  
   - Objectifs pédagogiques  

2. **Méthodes**  
   - Principe du perceptron simple  
   - Algorithme d’apprentissage  
   - Extension au multi-classe  

3. **Résultats**  
   - Tests sur fonctions logiques (ET, OU, XOR…)  
   - Analyse de la convergence (courbes d’erreur vs itérations)  
   - Évaluation sur données réelles  

4. **Discussion**  
   - Limites du perceptron (non-linéarité, sensibilité au jeu de données, choix du pas d’apprentissage…)  
   - Cas d’usage appropriés et alternatives (SVM, réseaux de neurones multilayer)  

5. **Conclusion**  
   - Synthèse des apprentissages  
   - Perspectives d’amélioration  

### 3. Visualisations

- **Graphiques de convergence**  
- **Visualisation des droites de séparation**  
- **Matrices de confusion**  
- **Comparaisons de performances**  

---


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



