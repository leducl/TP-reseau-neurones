import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

####### Exercice 4 - 5 - 6 ---------------------

# Classe du perceptron simple avec seuil de Heaviside (0 ou 1)
class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        """
        Initialisation du perceptron.
        learning_rate : pas d'apprentissage (float)
        weights       : vecteur de poids (numpy array), initialisé plus tard
        bias          : biais (float), initialisé à 0
        epochs_run    : nombre d'époques effectuées lors de l'entraînement (int)
        """
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0.0
        self.epochs_run = 0

    def fit(self, X, y, max_epochs=100):
        """
        Entraînement du perceptron sur les données X avec étiquettes y.
        X          : matrice des entrées (n_samples × n_features)
        y          : vecteur des sorties attendues (0 ou 1)
        max_epochs : nombre maximal d'époques
        """
        n_samples, n_features = X.shape
        # Initialisation aléatoire des poids, biais à zéro
        self.weights = np.random.randn(n_features)
        self.bias = 0.0

        # Boucle sur les époques avec barre de progression
        for epoch in tqdm(range(max_epochs), desc="Entraînement"):
            errors = 0  # compteur d'erreurs sur l'époque
            # Parcours de chaque exemple
            for xi, target in zip(X, y):
                # Calcul de l'activation linéaire
                activation = np.dot(self.weights, xi) + self.bias
                # Seuil de Heaviside : 1 si activation ≥ 0, sinon 0
                y_pred = 1 if activation >= 0 else 0

                # Si prédiction incorrecte, mise à jour des poids et du biais
                if y_pred != target:
                    # Calcul de l'erreur (target − prédiction)
                    update = self.learning_rate * (target - y_pred)
                    self.weights += update * xi    # ajustement des poids
                    self.bias    += update         # ajustement du biais
                    errors += 1

            # Si aucune erreur, arrêt anticipé
            if errors == 0:
                break

        # Stocke le nombre d'époques réellement effectuées
        self.epochs_run = epoch + 1

    def predict(self, X):
        """
        Prédiction sur un ensemble d'exemples X.
        Renvoie un vecteur d'entiers 0 ou 1.
        """
        # Calcul vectorisé des activations
        activations = np.dot(X, self.weights) + self.bias
        # Application du seuil : 1 si ≥0, sinon 0
        return (activations >= 0).astype(int)

    def score(self, X, y):
        """
        Calcul de l'exactitude (accuracy) : proportion de bonnes prédictions.
        """
        return np.mean(self.predict(X) == y)

    def tracer_decision(self, X, y, titre=""):
        """
        Affiche la frontière de décision et les points d'entraînement.
        X     : données d'entrée (2D uniquement)
        y     : étiquettes (0 ou 1)
        titre : titre du graphique (string)
        """
        # Définition de la grille de visualisation
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        points = np.c_[xx.ravel(), yy.ravel()]

        # Prédiction sur chaque point de la grille
        Z = self.predict(points).reshape(xx.shape)

        # Tracé de la région de décision
        plt.contourf(xx, yy, Z, alpha=0.3)
        # Tracé des points d'entraînement, couleur selon y
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        plt.title(titre)
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.show()


# -----------------------------------------
# Données logiques : AND, OR, XOR
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

X_or  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or  = np.array([0, 1, 1, 1])

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Test sur AND
p1 = PerceptronSimple()
p1.fit(X_and, y_and)
print("Score AND :", p1.score(X_and, y_and))
print("Époques AND :", p1.epochs_run)
p1.tracer_decision(X_and, y_and, "Pérceptron AND")

# Test sur OR
p2 = PerceptronSimple()
p2.fit(X_or, y_or)
print("Score OR :", p2.score(X_or, y_or))
print("Époques OR :", p2.epochs_run)
p2.tracer_decision(X_or, y_or, "Pérceptron OR")

# Test sur XOR (non linéairement séparable)
p3 = PerceptronSimple()
p3.fit(X_xor, y_xor)
print("Score XOR :", p3.score(X_xor, y_xor))
print("Époques XOR :", p3.epochs_run)
p3.tracer_decision(X_xor, y_xor, "Pérceptron XOR")
