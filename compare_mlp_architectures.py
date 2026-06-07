import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

ARCHITECTURES = [
    (64,),
    (128,),
    (256,),
    (128, 64),
    (256, 128),
    (512, 256),
    (256, 128, 64),
]


def label_architecture(arch):
    return "-".join(str(size) for size in arch)


def main():
    df = pd.read_csv("keypoints.csv")
    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    for arch in ARCHITECTURES:
        print(f"\nTraining MLP hidden_layer_sizes={arch}...")
        model = MLPClassifier(
            hidden_layer_sizes=arch,
            max_iter=500,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "architecture": label_architecture(arch),
            "hidden_layer_sizes": str(arch),
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "iterations": model.n_iter_,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "mlp_architecture_comparison.csv", index=False)
    print("\nResults:")
    print(result_df.to_string(index=False))

    plot_df = result_df.sort_values("accuracy", ascending=False)
    plt.figure(figsize=(10, 5.8))
    bars = plt.bar(plot_df["architecture"], plot_df["accuracy"] * 100, color="#147a58")
    plt.ylim(max(0, (plot_df["accuracy"].min() * 100) - 5), 100)
    plt.title("MLP Architecture Comparison on Yoga Pose Keypoints")
    plt.xlabel("Hidden layer sizes")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, plot_df["accuracy"] * 100):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mlp_architecture_accuracy.png", dpi=180)
    print(f"\nSaved {OUTPUT_DIR / 'mlp_architecture_comparison.csv'}")
    print(f"Saved {OUTPUT_DIR / 'mlp_architecture_accuracy.png'}")


if __name__ == "__main__":
    main()
