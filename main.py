"""Mini Project 2: Query By Committee for mushroom classification.

This script compares two active learning strategies:
1) Query By Committee (vote disagreement)
2) Random sampling baseline

The goal is to show that QBC can improve performance with fewer labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ActiveLearningConfig:
    """Configuration values for the active learning experiments."""

    dataset_path: Path = Path("Data/Mushroom data.csv")
    random_state: int = 42
    k_per_iteration: int = 10
    max_labeled_fraction: float = 0.30  # stop when 30% has been labeled
    excluded_features: tuple[str, ...] = ("odor",)
    dataset_fraction: float = 0.20  # use only 20% of rows to make task harder


def load_and_preprocess_data(
    dataset_path: Path,
    excluded_features: tuple[str, ...] = (),
    dataset_fraction: float = 1.0,
    random_state: int = 42,
):
    """Load mushroom CSV, inspect features, one-hot encode features, and encode labels.

    Label mapping:
    - p (poisonous) -> 1
    - e (edible)    -> 0
    """
    df = pd.read_csv(dataset_path)

    if not (0 < dataset_fraction <= 1.0):
        raise ValueError("dataset_fraction must be in the range (0, 1].")

    # The dataset usually stores class label in column named 'class'.
    # We keep a fallback to first column to make this robust.
    label_column = "class" if "class" in df.columns else df.columns[0]

    # Print included features so it is easy to verify requirement #1.
    feature_columns = [col for col in df.columns if col != label_column]

    # Optionally remove selected features (for controlled experiments).
    excluded_set = set(excluded_features)
    if excluded_set:
        feature_columns = [col for col in feature_columns if col not in excluded_set]
    print("Features included in dataset:")
    for feature in feature_columns:
        print(f"- {feature}")

    if excluded_set:
        print("\nFeatures excluded from this run:")
        for feature in sorted(excluded_set):
            print(f"- {feature}")

    y_raw = df[label_column]
    X_raw = df[feature_columns]

    # Optionally reduce the amount of data to make the experiment harder.
    if dataset_fraction < 1.0:
        sampled_indices = y_raw.groupby(y_raw, group_keys=False).apply(
            lambda group: group.sample(
                frac=dataset_fraction,
                random_state=random_state,
            )
        ).index
        X_raw = X_raw.loc[sampled_indices]
        y_raw = y_raw.loc[sampled_indices]

        print(
            f"\nUsing reduced dataset fraction: {dataset_fraction:.2f} "
            f"({len(y_raw)} rows)"
        )

    # One-hot encoding for all categorical features.
    X = pd.get_dummies(X_raw, drop_first=False)

    # Encode labels to binary values.
    y = y_raw.map({"e": 0, "p": 1})
    if y.isna().any():
        raise ValueError("Unexpected labels detected. Expected only 'e' and 'p'.")

    return X.values, y.values.astype(int), feature_columns


def create_initial_splits(X, y, random_state: int):
    """Split data into L (10%), U (80%), T (10%)."""
    n_samples = X.shape[0]
    all_indices = np.arange(n_samples)

    # First split off test set T (10%).
    train_pool_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.10,
        random_state=random_state,
        stratify=y,
    )

    # From remaining 90%, take 1/9 for L so L becomes 10% of total.
    # 90% * (1/9) = 10%
    labeled_idx, unlabeled_idx = train_test_split(
        train_pool_idx,
        test_size=8 / 9,
        random_state=random_state,
        stratify=y[train_pool_idx],
    )

    return labeled_idx, unlabeled_idx, test_idx


def initialize_committee(random_state: int):
    """Create exactly the 3 required committee models."""
    return [
        LogisticRegression(max_iter=1_000, random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        GaussianNB(),
    ]


def compute_vote_disagreement(committee, X_pool):
    """Compute vote-based disagreement for each sample in pool.

    For binary labels with 3 models, disagreement is the minority vote count,
    which is:
    - 0 for unanimous votes (3-0)
    - 1 for split votes (2-1)
    """
    # shape: (n_models, n_pool_samples)
    all_predictions = np.vstack([model.predict(X_pool) for model in committee])

    votes_for_class_1 = all_predictions.sum(axis=0)
    votes_for_class_0 = committee.__len__() - votes_for_class_1

    disagreement = np.minimum(votes_for_class_0, votes_for_class_1)
    return disagreement


def evaluate_committee(committee, X_test, y_test):
    """Evaluate committee by majority vote and return accuracy + f1."""
    preds = np.vstack([model.predict(X_test) for model in committee])
    majority_vote = (preds.sum(axis=0) >= 2).astype(int)

    return {
        "accuracy": accuracy_score(y_test, majority_vote),
        "f1": f1_score(y_test, majority_vote),
    }


def run_active_learning(
    X,
    y,
    labeled_idx_start,
    unlabeled_idx_start,
    test_idx,
    config: ActiveLearningConfig,
    strategy: str,
):
    """Run active learning loop using either 'qbc' or 'random'."""
    labeled_idx = labeled_idx_start.copy()
    unlabeled_idx = unlabeled_idx_start.copy()

    n_total = len(y)
    budget_count = int(config.max_labeled_fraction * n_total)

    labeled_counts = []
    accuracies = []
    f1_scores = []

    rng = np.random.default_rng(config.random_state)

    iteration = 0
    while len(labeled_idx) < budget_count and len(unlabeled_idx) > 0:
        iteration += 1

        committee = initialize_committee(config.random_state)

        # Train all models on currently labeled set L.
        for model in committee:
            model.fit(X[labeled_idx], y[labeled_idx])

        # Evaluate on test set T.
        metrics = evaluate_committee(committee, X[test_idx], y[test_idx])
        labeled_counts.append(len(labeled_idx))
        accuracies.append(metrics["accuracy"])
        f1_scores.append(metrics["f1"])

        if strategy == "qbc":
            disagreement = compute_vote_disagreement(committee, X[unlabeled_idx])
            top_k = min(config.k_per_iteration, len(unlabeled_idx))
            query_order = np.argsort(-disagreement)
            selected_positions = query_order[:top_k]
        elif strategy == "random":
            top_k = min(config.k_per_iteration, len(unlabeled_idx))
            selected_positions = rng.choice(len(unlabeled_idx), size=top_k, replace=False)
        else:
            raise ValueError("strategy must be either 'qbc' or 'random'")

        selected_indices = unlabeled_idx[selected_positions]

        # Move selected from U -> L and reveal true labels (simulated annotation).
        labeled_idx = np.concatenate([labeled_idx, selected_indices])
        unlabeled_idx = np.delete(unlabeled_idx, selected_positions)

        print(
            f"[{strategy.upper()}] Iteration {iteration:02d} | "
            f"L={len(labeled_idx):4d}, U={len(unlabeled_idx):4d}, "
            f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}"
        )

    # Final train/eval after loop for final report.
    final_committee = initialize_committee(config.random_state)
    for model in final_committee:
        model.fit(X[labeled_idx], y[labeled_idx])
    final_metrics = evaluate_committee(final_committee, X[test_idx], y[test_idx])

    history = {
        "labeled_counts": labeled_counts,
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "final_metrics": final_metrics,
    }
    return history


def plot_learning_curves(qbc_history, random_history):
    """Plot accuracy vs number of labeled samples for QBC and random baseline."""
    plt.figure(figsize=(9, 6))
    plt.plot(
        qbc_history["labeled_counts"],
        qbc_history["accuracies"],
        marker="o",
        label="QBC (vote disagreement)",
    )
    plt.plot(
        random_history["labeled_counts"],
        random_history["accuracies"],
        marker="s",
        label="Random sampling",
    )
    plt.xlabel("Number of labeled samples")
    plt.ylabel("Test accuracy")
    plt.title("Active Learning: QBC vs Random Sampling")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    config = ActiveLearningConfig()

    X, y, _ = load_and_preprocess_data(
        config.dataset_path,
        excluded_features=config.excluded_features,
        dataset_fraction=config.dataset_fraction,
        random_state=config.random_state,
    )

    labeled_idx, unlabeled_idx, test_idx = create_initial_splits(
        X, y, config.random_state
    )

    print(
        f"\nInitial split sizes -> L: {len(labeled_idx)}, "
        f"U: {len(unlabeled_idx)}, T: {len(test_idx)}"
    )

    qbc_history = run_active_learning(
        X,
        y,
        labeled_idx_start=labeled_idx,
        unlabeled_idx_start=unlabeled_idx,
        test_idx=test_idx,
        config=config,
        strategy="qbc",
    )

    random_history = run_active_learning(
        X,
        y,
        labeled_idx_start=labeled_idx,
        unlabeled_idx_start=unlabeled_idx,
        test_idx=test_idx,
        config=config,
        strategy="random",
    )

    print("\nFinal evaluation on test set (majority vote across committee):")
    print(
        f"QBC    -> Accuracy: {qbc_history['final_metrics']['accuracy']:.4f}, "
        f"F1: {qbc_history['final_metrics']['f1']:.4f}"
    )
    print(
        f"Random -> Accuracy: {random_history['final_metrics']['accuracy']:.4f}, "
        f"F1: {random_history['final_metrics']['f1']:.4f}"
    )

    plot_learning_curves(qbc_history, random_history)


if __name__ == "__main__":
    main()
