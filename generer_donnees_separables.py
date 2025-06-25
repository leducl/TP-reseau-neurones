import numpy as np
import matplotlib.pyplot as plt
from PerceptronSimple import PerceptronSimple  # import de la classe perceptron

####### Exercice 7 ---------------------


# fonction pour generer des donnees lineairement separables
def generer_donnees_separables(n_points=100, noise=0.1):
    np.random.seed(42)  # pour avoir toujours les memes resultats

    # classe -1 autour du point (-2, -2)
    X_neg = np.random.randn(n_points // 2, 2) * noise + np.array([-2, -2])
    y_neg = -np.ones(n_points // 2)  # tous les labels a -1

    # classe +1 autour du point (2, 2)
    X_pos = np.random.randn(n_points // 2, 2) * noise + np.array([2, 2])
    y_pos = np.ones(n_points // 2)  # tous les labels a +1

    # on regroupe les deux classes ensemble
    X = np.vstack((X_neg, X_pos))  # les coordonnees
    y = np.concatenate((y_neg, y_pos))  # les labels

    return X, y  # on retourne les donnees

# fonction pour afficher les donnees et la droite de separation si dispo
def visualiser_donnees(X, y, w=None, b=None, title="Donnees"):
    plt.figure(figsize=(8, 6))

    # on separe les points selon leur classe
    mask_pos = (y == 1)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c='blue', marker='+', s=100, label='classe +1')
    plt.scatter(X[~mask_pos, 0], X[~mask_pos, 1], c='red', marker='*', s=100, label='classe -1')

    # on trace la droite w·x + b = 0 si on a les valeurs
    if w is not None and b is not None:
        if w[1] != 0:  # cas normal
            x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
            y_vals = -(w[0] * x_vals + b) / w[1]
            plt.plot(x_vals, y_vals, 'k--', label='frontiere de decision')
        else:
            # cas special ou la droite est verticale
            x_vert = -b / w[0]
            plt.axvline(x=x_vert, color='k', linestyle='--', label='frontiere de decision')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ---------------------
# exemple d'utilisation
# ---------------------

# on genere les donnees
X, y = generer_donnees_separables(n_points=100, noise=0.1)

# on affiche les donnees avant l'apprentissage
visualiser_donnees(X, y, title="donnees generees (avant apprentissage)")

# on cree et entraine le perceptron
perceptron = PerceptronSimple(learning_rate=0.1)
perceptron.fit(X, y)

# on affiche le score d'apprentissage
print(f"score du perceptron : {perceptron.score(X, y) * 100:.2f}%")

# on affiche les donnees avec la droite apprise
visualiser_donnees(X, y, w=perceptron.weights, b=perceptron.bias, title="donnees avec frontiere apprise")
