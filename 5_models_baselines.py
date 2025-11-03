
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from config import paths, settings
from utils_io import save_figure, save_table

def load_cache():
    X = np.load(os.path.join(paths.cache_dir, "X_merged.npy"))
    y_path = os.path.join(paths.cache_dir, "y.npy")
    y = np.load(y_path) if os.path.exists(y_path) else None
    return X, y

def train_eval_baselines():
    X, y = load_cache()
    if y is None:
        print("labels not found, skipping baselines")
        return

    models = {
        "svm_rbf": SVC(kernel="rbf", probability=True, C=2.0, gamma="scale", random_state=settings.random_seed),
        "ffnn_mlp": MLPClassifier(hidden_layer_sizes=(256,128), activation="relu", alpha=1e-4,
                                  learning_rate_init=1e-3, max_iter=100, random_state=settings.random_seed),
        "logreg": LogisticRegression(max_iter=200)
    }
    skf = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed)
    rows = []
    for name, model in models.items():
        y_true_all, y_prob_all = [], []
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            model.fit(X[tr], y[tr])
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X[va])[:,1]
            else:
                # fallback score to decision function
                prob = model.decision_function(X[va])
                # scale to 0..1
                prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
            pred = (prob >= 0.5).astype(int)
            y_true_all.extend(y[va].tolist())
            y_prob_all.extend(prob.tolist())

        y_true_all = np.array(y_true_all)
        y_prob_all = np.array(y_prob_all)
        auc = roc_auc_score(y_true_all, y_prob_all) if len(np.unique(y_true_all))>1 else np.nan
        acc = accuracy_score(y_true_all, (y_prob_all>=0.5).astype(int))
        rows.append({"model": name, "auc": auc, "accuracy": acc})

        # ROC plot
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_true_all, y_prob_all, ax=ax, name=name)
        ax.set_title(f"ROC {name}")
        save_figure(fig, f"roc_{name}")
        plt.close(fig)

        # Confusion matrix
        cm = confusion_matrix(y_true_all, (y_prob_all>=0.5).astype(int))
        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_title(f"Confusion matrix {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center")
        save_figure(fig, f"cm_{name}")
        plt.close(fig)

    df = pd.DataFrame(rows)
    save_table(df, "baseline_metrics")

if __name__ == "__main__":
    train_eval_baselines()
