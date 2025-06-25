from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def charger_donnees_iris_binaire():
    """
    Charge le dataset Iris en version binaire (2 classes) pour commencer
    """
    # Chargement du dataset Iris
    iris = load_iris()

    # Ne garder que 2 features pour la visualisation
    X = iris.data[:, [0, 2]]  # longueur des sépales et longueur des pétales
    y = iris.target

    # Ne garder que 2 classes pour commencer (Setosa vs Versicolor)
    mask = y < 2
    X_binary = X[mask]
    y_binary = y[mask]
    y_binary = 2 * y_binary - 1  # Convertir 0,1 en -1,1

    return X_binary, y_binary

def charger_donnees_iris_complete():
    """
    Charge le dataset Iris complet avec les 3 classes
    """
    iris = load_iris()
    X = iris.data[:, [0, 2]]  # longueur des sépales et longueur des pétales
    y = iris.target

    return X, y, iris.target_names

def visualiser_iris(X, y, target_names=None, title="Dataset Iris"):
    """
    Visualise le dataset Iris avec ses différentes classes
    """
    plt.figure(figsize=(10, 8))

    # Couleurs pour chaque classe
    colors = ['red', 'blue', 'green']
    markers = ['*', '+', 'o']

    for i in range(len(np.unique(y))):
        mask = (y == i)
        label = target_names[i] if target_names else f'Classe {i}'
        plt.scatter(X[mask, 0], X[mask, 1],
                   c=colors[i], marker=markers[i], s=100,
                   label=label, alpha=0.7)

    plt.xlabel('Longueur des sépales (cm)')
    plt.ylabel('Longueur des pétales (cm)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Test de la fonction
if __name__ == "__main__":
    # Version binaire d'abord
    X_bin, y_bin = charger_donnees_iris_binaire()
    print(f"Données binaires : {X_bin.shape[0]} échantillons, {X_bin.shape[1]} features")

    # Version complète
    X_full, y_full, noms = charger_donnees_iris_complete()
    print(f"Données complètes : {X_full.shape[0]} échantillons, {len(np.unique(y_full))} classes")

    # Visualisation
    visualiser_iris(X_full, y_full, noms)