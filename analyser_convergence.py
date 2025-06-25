import numpy as np
import matplotlib.pyplot as plt


####### Exercice 8 ---------------------

# fonction pour analyser la convergence du perceptron selon différents eta
def analyser_convergence(X, y, learning_rates=[0.001], max_epochs=30):

    plt.figure(figsize=(12, 8))

    # On teste chaque taux d'apprentissage
    for lr in learning_rates:
        # Initialisation aléatoire des poids et biais
        n_features = X.shape[1]
        weights = np.random.randn(n_features)  # poids initiaux
        bias = 0.0  # biais initial

        erreurs_par_epoque = []  # pour stocker le nb d'erreurs à chaque époque

        # Boucle d'entraînement
        for epoch in range(1, max_epochs + 1):
            erreurs = 0  # compteur d'erreurs pour cette époque

            # Parcours de chaque exemple
            for xi, target in zip(X, y):
                # Calcul de l'activation linéaire
                activation = np.dot(weights, xi) + bias
                # Seuil de Heaviside : 1 si activation ≥ 0, sinon 0
                pred = 1 if activation >= 0 else 0

                # Si prédiction incorrecte, mise à jour des poids et du biais
                if pred != target:
                    # Calcul de la correction (target − prédiction)
                    update = lr * (target - pred)
                    weights += update * xi  # ajustement des poids
                    bias += update  # ajustement du biais
                    erreurs += 1

            # Stocke le nombre d'erreurs pour cette époque
            erreurs_par_epoque.append(erreurs)

        # Trace la courbe erreurs vs époque pour ce learning rate
        plt.plot(
            range(1, max_epochs + 1),
            erreurs_par_epoque,
            marker='o',
            label=f'η = {lr}'
        )

    # Mise en forme du graphique
    plt.xlabel('Époque')
    plt.ylabel('Nombre d’erreurs')
    plt.title('Convergence du perceptron (Heaviside) pour différents η')
    plt.legend(title='Learning rate')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------
# Code principal, exécuté si lancé en script
if __name__ == '__main__':
    np.random.seed(0)  # pour reproductibilité

    # Génération de données bruitées
    n = 200  # nombre total de points
    noise = 0.1 # niveau de bruit

    # points négatifs autour de (-2, -2) => étiquette 0
    X_neg = np.random.randn(n // 2, 2) * noise + np.array([-2, -2])
    y_neg = np.zeros(n // 2, dtype=int)

    # points positifs autour de (2, 2) => étiquette 1
    X_pos = np.random.randn(n // 2, 2) * noise + np.array([2, 2])
    y_pos = np.ones(n // 2, dtype=int)

    # Combinaison et mélange des données
    X = np.vstack([X_neg, X_pos])
    y = np.hstack([y_neg, y_pos])
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]


    analyser_convergence(X, y, max_epochs=100)
