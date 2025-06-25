import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PerceptronMultiClasse import PerceptronMultiClasse

# on definit heaviside comme fonction d activation
def heaviside(z):
    return np.where(z >= 0, 1, 0)

def evaluer_perceptron_multiclasse(
    X, y,
    target_names=None,
    test_size=0.3,
    val_size=0.5,
    learning_rate=0.1,
    max_epochs=100,
    patience=10
):
    # on separe d abord train+val et test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    # on separe ensuite train et validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=42,
        stratify=y_temp
    )

    # on affiche la repartition des donnees
    print("repartition des donnees :")
    print(f"  - entrainement : {X_train.shape[0]} echantillons")
    print(f"  - validation   : {X_val.shape[0]} echantillons")
    print(f"  - test         : {X_test.shape[0]} echantillons\n")

    # on instancie le perceptron
    perceptron_mc = PerceptronMultiClasse(learning_rate=learning_rate)
    best_val_acc = 0.0
    best_weights = None
    epochs_no_improve = 0

    # boucle d entrainement avec early stopping
    for epoch in range(1, max_epochs + 1):
        # on entraine une epoch
        perceptron_mc.partial_fit(X_train, y_train)
        # on calcule la sortie lineaire sur validation
        z_val = X_val.dot(perceptron_mc.weights.T) + perceptron_mc.bias
        # on applique heaviside pour avoir y pred
        y_val_pred = np.argmax(heaviside(z_val), axis=1)
        val_acc = np.mean(y_val_pred == y_val)

        # on verifie si c est la meilleure accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = perceptron_mc.get_weights().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # si pas d amelioration assez longtemps on arrete
        if epochs_no_improve >= patience:
            print(f"early stopping a l epoch {epoch}")
            break

    # on restaure les meilleurs poids
    perceptron_mc.set_weights(best_weights)

    # on predit pour train validation test avec heaviside
    z_train = X_train.dot(perceptron_mc.weights.T) + perceptron_mc.bias
    z_val   = X_val.dot(perceptron_mc.weights.T)   + perceptron_mc.bias
    z_test  = X_test.dot(perceptron_mc.weights.T)  + perceptron_mc.bias

    y_train_pred = np.argmax(heaviside(z_train), axis=1)
    y_val_pred   = np.argmax(heaviside(z_val),   axis=1)
    y_test_pred  = np.argmax(heaviside(z_test),  axis=1)

    # on calcule les metrics
    results = {}
    for split, y_true, y_pred in [
        ("train", y_train, y_train_pred),
        ("val",   y_val,   y_val_pred),
        ("test",  y_test,  y_test_pred)
    ]:
        acc = np.mean(y_pred == y_true)
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        results[split] = {
            "accuracy": acc,
            "report": report,
            "cm": cm
        }

    # on affiche les accuracies
    print("performances :")
    for split in ["train", "val", "test"]:
        print(f"  - accuracy {split:9s}: {results[split]['accuracy']:.3f}")
    # on affiche le rapport de classification pour le test
    print("\nrapport de classification test :")
    print(results["test"]["report"])

    # on trace les matrices de confusion
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, split in zip(axes, ["train", "val", "test"]):
        cm = results[split]["cm"]
        sns.heatmap(
            cm,
            annot=True, fmt="d",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"cm {split}")
        ax.set_xlabel("predit")
        ax.set_ylabel("reel")
    plt.tight_layout()
    plt.show()

    return results

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    def generate_data_lin_sep():
        # 3 classes bien séparées
        centres = np.array([[0, 0], [3, 3], [0, 4]])
        X, y = [], []
        for i, centre in enumerate(centres):
            X.append(np.random.randn(50, 2) * 0.5 + centre)
            y += [i]*50
        return np.vstack(X), np.array(y)

    def generate_data_xor():
        # XOR étendu à 3 classes en 2D
        X = np.random.randn(300, 2)
        y = np.array([0 if x[0]*x[1] >= 0 else 1 for x in X])
        y[::5] = 2  # on ajoute une 3e classe pour brouiller
        return X, y

    def plot_data(X, y, title="Données"):
        plt.figure()
        for label in np.unique(y):
            plt.scatter(X[y==label, 0], X[y==label, 1], label=f"Classe {label}")
        plt.title(title)
        plt.legend()
        plt.show()

    def test_perceptron(description, X, y, normalize=False, imbalance=False, noise=0.0):
        print(f"\n=== {description} ===")

        if imbalance:
            # on déséquilibre les classes
            mask = (y != 0) | (np.random.rand(len(y)) < 0.2)
            X, y = X[mask], y[mask]

        if noise > 0:
            X += np.random.normal(0, noise, X.shape)

        if normalize:
            X = StandardScaler().fit_transform(X)

        # séparation
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

        # entraînement
        model = PerceptronMultiClasse(learning_rate=0.05)
        model.fit(X_train, y_train, max_epochs=100)

        # prédiction et évaluation
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        print(f"Accuracy test: {acc:.2f}")
        print(classification_report(y_test, y_pred, zero_division=0))

    np.random.seed(42)

    # 1. Données linéairement séparables
    X1, y1 = generate_data_lin_sep()
    test_perceptron("Convergence sur données linéairement séparables", X1, y1)

    # 2. Test avec normalisation
    test_perceptron("Avec normalisation", X1, y1, normalize=True)

    # 3. Données bruitées
    test_perceptron("Données bruitées", X1, y1, noise=1.0)

    # 4. Données déséquilibrées
    test_perceptron("Classes déséquilibrées", X1, y1, imbalance=True)

    # 5. Cas XOR
    X2, y2 = generate_data_xor()
    test_perceptron("Cas XOR (non séparables linéairement)", X2, y2)

    # 6. Test avec différents taux d'apprentissage
    print("\nTest taux d'apprentissage :")
    for lr in [0.001, 0.01, 0.1, 1.0]:
        print(f" - lr={lr}")
        model = PerceptronMultiClasse(learning_rate=lr)
        model.fit(X1, y1, max_epochs=50)
        y_pred = model.predict(X1)
        acc = np.mean(y_pred == y1)
        print(f"   Accuracy train: {acc:.2f}")

    # 7. Sensibilité à l'initialisation
    print("\nTest sensibilité à l'initialisation :")
    for i in range(3):
        np.random.seed(i)
        model = PerceptronMultiClasse(learning_rate=0.05)
        model.fit(X1, y1, max_epochs=50)
        acc = np.mean(model.predict(X1) == y1)
        print(f"  Essai {i+1}, accuracy train: {acc:.2f}")
