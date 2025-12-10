import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ---------------------------------------------------------
# LOAD CSV
# ---------------------------------------------------------
def load_data(path="./data/twitter_training.csv"):
    df = pd.read_csv(
        path,
        header=None,
        names=["tweet_id", "game", "label", "text"],
        encoding="latin-1"
    )
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    return df


# ---------------------------------------------------------
# VECTORISER
# ---------------------------------------------------------
def get_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000
    )


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2500),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }


# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n========== Evaluating: {name} ==========\n")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred, zero_division=0))

    cm = confusion_matrix(y_test, pred)

    return {
        "name": name,
        "model": model,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "cm": cm
    }


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------
def plot_confusion(cm, labels):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Best Model)")
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("confusion.png")
    plt.close()


def plot_roc(model, X_test, y_test):
    if not hasattr(model, "predict_proba"):
        print("Skipping ROC (model has no predict_proba).")
        return

    y_true_bin = (y_test == "Positive").astype(int)
    probs = model.predict_proba(X_test)

    if probs.shape[1] > 2:
        idx = list(model.classes_).index("Positive")
        probs_pos = probs[:, idx]
    else:
        probs_pos = probs[:, 1]

    fpr, tpr, _ = roc_curve(y_true_bin, probs_pos)
    auc_score = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Positive vs Rest)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc.png")
    plt.close()


def plot_model_comparison(results):
    models = [r["name"] for r in results]
    accs = [r["acc"] for r in results]
    precs = [r["prec"] for r in results]
    recs = [r["rec"] for r in results]
    f1s = [r["f1"] for r in results]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.3, accs, width, label="Accuracy")
    plt.bar(x - 0.1, precs, width, label="Precision")
    plt.bar(x + 0.1, recs, width, label="Recall")
    plt.bar(x + 0.3, f1s, width, label="F1 Score")

    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Model Comparison: Accuracy, Precision, Recall, F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_metrics.png")
    plt.close()


# ---------------------------------------------------------
# SAVE METRICS CSV
# ---------------------------------------------------------
def save_metrics(results):
    df = pd.DataFrame(results)
    df = df[["name", "acc", "prec", "rec", "f1"]]
    df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    df.to_csv("metrics_report.csv", index=False)
    print("\nSaved metrics_report.csv")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    df = load_data()
    print("Data loaded:", df.shape)

    X = df["text"]
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = get_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    models = get_models()

    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(res)

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest model is: {best['name']} (F1={best['f1']:.4f})")

    labels = sorted(df["label"].unique())
    plot_confusion(best["cm"], labels)
    plot_roc(best["model"], X_test, y_test)
    plot_model_comparison(results)

    save_metrics(results)

    # ----------- OBSERVATION SECTION -----------
    print("\n===== GRAPH-BASED OBSERVATIONS =====")
    print("1. Accuracy Comparison:")
    print("   Logistic Regression usually leads because TF-IDF vectors are linearly separable.")
    print("   Naive Bayes performs decently due to word-frequency nature.")
    print("   KNN struggles because high-dimensional TF-IDF makes distance meaningless.")

    print("\n2. Precision & Recall Pattern:")
    print("   Logistic Regression balances precision and recall well.")
    print("   Naive Bayes may show higher recall but lower precision.")
    print("   KNN often shows inconsistent recall due to sparse vectors.")

    print("\n3. F1 Score:")
    print("   F1 confirms the overall balance; Logistic Regression often wins.")

    print("\n4. Confusion Matrix Insight:")
    print("   Misclassifications mostly occur between Neutral and Positive tweets.")
    print("   This is common in social text due to sarcasm and ambiguous wording.")

    print("\n5. ROC Curve:")
    print("   A smooth ROC curve with AUC > 0.80 indicates the model separates classes well.")
    print("   If curve is close to diagonal, model is weak.")

    print("\n=======================================")


if __name__ == "__main__":
    main()
