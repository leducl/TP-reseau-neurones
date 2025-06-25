import numpy as np
import matplotlib.pyplot as plt


####### Exercice 1 ---------------------


class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha  # pour leaky relu qd z negatif

    def apply(self, z):
        if self.name == "heaviside":
            return np.where(z >= 0, 1, -1)  # sort 1 si positif sinon -1
        elif self.name == "sigmoid":
            return 1 / (1 + np.exp(-z))  # fonction en S tres utilisee
        elif self.name == "tanh":
            return np.tanh(z)  # hyperbolique tangent entre -1 et 1
        elif self.name == "relu":
            return np.maximum(0, z)  # garde les pos sinon 0
        elif self.name == "leaky_relu":
            return np.where(z >= 0, z, self.alpha * z)  # comme relu mais laisse un peu passer les neg
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue")  # cas erreur si nom mauvais

    def derivative(self, z):
        if self.name == "heaviside":
            return np.zeros_like(z)  # pas derivee def pour heaviside
        elif self.name == "sigmoid":
            sig = self.apply(z)
            return sig * (1 - sig)  # derivee classique sigmoid
        elif self.name == "tanh":
            return 1 - np.tanh(z) ** 2  # derivee tanh
        elif self.name == "relu":
            return np.where(z > 0, 1, 0)  # 1 si z > 0 sinon 0
        elif self.name == "leaky_relu":
            return np.where(z >= 0, 1, self.alpha)  # pareil que relu mais alpha pour les neg
        else:
            raise ValueError(f"derivee de '{self.name}' pas definie")  # message si erreur

# generation des valeurs z a tester
z = np.linspace(-5, 5, 400)

# liste des activations a tester
activations = ['heaviside', 'sigmoid', 'tanh', 'relu', 'leaky_relu']

# creation des subplots pour afficher
fig, axes = plt.subplots(len(activations), 2, figsize=(12, 18))

for i, name in enumerate(activations):
    af = ActivationFunction(name)
    axes[i, 0].plot(z, af.apply(z))  # plot la fonction
    axes[i, 0].set_title(f"{name} - Activation")  # titre du graphique
    axes[i, 0].grid(True)

    axes[i, 1].plot(z, af.derivative(z))  # plot la derivee
    axes[i, 1].set_title(f"{name} - Dérivée")  # titre derivee
    axes[i, 1].grid(True)

plt.tight_layout()
plt.show()  # affiche tout
