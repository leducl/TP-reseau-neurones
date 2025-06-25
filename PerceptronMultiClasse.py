import numpy as np
from PerceptronSimple import PerceptronSimple  # perceptron binaire deja implemente

class PerceptronMultiClasse:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.perceptrons = {}  # dictionnaire classe -> perceptron
        self.classes = None

    def fit(self, X, y, max_epochs=100):
        """
        entraine un perceptron par classe (un-contre-tous)
        """
        # recuperer liste des classes uniques
        self.classes = np.unique(y)

        # pour chaque classe, creer et entrainer un perceptron binaire
        for classe in self.classes:
            # construire targets binaires : +1 si appartient a la classe, -1 sinon
            y_binary = np.where(y == classe, 1, -1)
            # creer perceptron simple
            percep = PerceptronSimple(learning_rate=self.learning_rate)
            # entrainer sur le probleme binaire
            percep.fit(X, y_binary, max_epochs)
            # stocker
            self.perceptrons[classe] = percep

    def predict(self, X):
        """
        predit la classe avec le score le plus eleve
        """
        if not self.perceptrons:
            raise ValueError("modele non entraine, appelez fit() d abord.")

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        # tableau pour stocker score brut de chaque perceptron
        scores = np.zeros((n_samples, n_classes))

        # calculer score brut pour chaque classe
        for idx, classe in enumerate(self.classes):
            percep = self.perceptrons[classe]
            # produit scalaire + biais
            raw = X.dot(percep.weights) + percep.bias
            scores[:, idx] = raw

        # choisir la classe dont le score est maximum
        choix = np.argmax(scores, axis=1)
        return self.classes[choix]

    def predict_proba(self, X):
        """
        retourne les scores bruts (sans normalisation)
        """
        if not self.perceptrons:
            raise ValueError("modele non entraine, appelez fit() d abord.")

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        scores = np.zeros((n_samples, n_classes))

        for idx, classe in enumerate(self.classes):
            percep = self.perceptrons[classe]
            scores[:, idx] = X.dot(percep.weights) + percep.bias

        return scores

# test basique du perceptron multi-classes
if __name__ == '__main__':
    np.random.seed(1)
    # generation de donnees pour 3 classes
    n_samples = 150
    centres = np.array([[0, 0], [3, 3], [0, 4]])
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    for i, centre in enumerate(centres):
        X[i*50:(i+1)*50] = np.random.randn(50, 2) * 0.5 + centre
        y[i*50:(i+1)*50] = i
    # melange et separation train / test
    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]
    split = int(0.7 * n_samples)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    # creation et entrainement
    model = PerceptronMultiClasse(learning_rate=0.1)
    model.fit(X_train, y_train, max_epochs=50)
    # prediction et evaluation
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"accuracy test multi-classe: {acc*100:.2f}%")


