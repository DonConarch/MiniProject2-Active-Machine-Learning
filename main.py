import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# 1. Data Handling
# ============================================================

# Load the spambase dataset
df = pd.read_csv("data/spambase.csv")

print("Shape:", df.shape)
print("\nLabel distribution:")
print(df["spam"].value_counts())

# Features and labels
X = df.drop("spam", axis=1).values.astype(float)
y = df["spam"].values.astype(float)  # 0 = not spam, 1 = spam

# Standardize features (important for Logistic Regression)
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"\nFeature matrix shape: {X.shape}")

# ============================================================
# 2. Committee Models
# ============================================================

def create_committee():
    """Create a fresh committee of 3 diverse classifiers."""
    return [
        LogisticRegression(max_iter=1000, random_state=42),
        DecisionTreeClassifier(random_state=42),
        GaussianNB()
    ]

# ============================================================
# 3. Disagreement Function
# ============================================================

def compute_disagreement(committee, X_unlabeled):
    """
    Compute vote-based disagreement for each sample in the pool.
    For each sample, count how many models disagree on the predicted label.
    Disagreement = 1 - (max_votes / num_models)
    A sample with full agreement gets 0, maximum disagreement gets higher values.
    """
    # Collect predictions from each model
    predictions = np.array([model.predict(X_unlabeled) for model in committee])
    # predictions shape: (n_models, n_samples)

    n_samples = X_unlabeled.shape[0]
    disagreement = np.zeros(n_samples)

    for i in range(n_samples):
        votes = predictions[:, i]
        # Count votes for the most common label
        most_common_count = max(np.sum(votes == 0), np.sum(votes == 1))
        # Disagreement: fraction of models that disagree with the majority
        disagreement[i] = 1 - (most_common_count / len(committee))

    return disagreement

# ============================================================
# 4. Active Learning Loop (QBC + Random baseline)
# ============================================================

def run_active_learning(X, y, X_test, y_test, strategy="qbc", k=10,
                        max_fraction=0.30, seed=0):
    """
    Run one active learning experiment.

    Parameters:
        strategy: "qbc" for Query By Committee, "random" for random sampling
        k: number of samples to query per iteration
        max_fraction: stop when this fraction of total data is labeled
        seed: random seed for reproducibility

    Returns:
        n_labeled_list: list of labeled set sizes at each step
        accuracy_list: test accuracy at each step
        f1_list: test F1-score at each step
    """
    rng = np.random.RandomState(seed)

    # Split into L (10%), U (80%), T (10%) with this seed
    X_temp, _, y_temp, _ = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=y
    )
    X_labeled, X_pool, y_labeled, y_pool = train_test_split(
        X_temp, y_temp, test_size=8/9, random_state=seed, stratify=y_temp
    )

    # Convert to lists for easy appending
    X_labeled = list(X_labeled)
    y_labeled = list(y_labeled)
    X_pool = list(X_pool)
    y_pool = list(y_pool)

    # Track performance
    n_labeled_list = []
    accuracy_list = []
    f1_list = []

    max_labeled = int(max_fraction * len(y))

    while len(X_labeled) <= max_labeled and len(X_pool) > 0:
        # --- Train committee on current labeled set ---
        X_L = np.array(X_labeled)
        y_L = np.array(y_labeled)
        committee = create_committee()
        for model in committee:
            model.fit(X_L, y_L)

        # --- Evaluate on test set (use majority vote of committee) ---
        test_predictions = np.array([model.predict(X_test) for model in committee])
        # Majority vote: round the mean prediction
        majority_vote = (np.mean(test_predictions, axis=0) >= 0.5).astype(float)

        acc = accuracy_score(y_test, majority_vote)
        f1 = f1_score(y_test, majority_vote)

        n_labeled_list.append(len(X_labeled))
        accuracy_list.append(acc)
        f1_list.append(f1)

        # --- Select samples to query ---
        X_U = np.array(X_pool)

        if len(X_pool) <= k:
            # Not enough samples left, take all
            break

        if strategy == "qbc":
            # Use vote disagreement to pick the most uncertain samples
            disagreement = compute_disagreement(committee, X_U)
            query_indices = np.argsort(disagreement)[-k:]  # top-k most disagreed
        else:
            # Random baseline: pick k random samples
            query_indices = rng.choice(len(X_pool), size=k, replace=False)

        # --- Move queried samples from pool to labeled set ---
        # Sort indices in reverse so removing doesn't shift later indices
        query_indices_sorted = sorted(query_indices, reverse=True)
        for idx in query_indices_sorted:
            X_labeled.append(X_pool.pop(idx))
            y_labeled.append(y_pool.pop(idx))

    return n_labeled_list, accuracy_list, f1_list

# ============================================================
# 5. Run Experiments (averaged over multiple runs)
# ============================================================

# Fixed test set for fair comparison across all runs
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

n_runs = 20  # Number of runs to average over
k = 10       # Samples queried per iteration

print("\nRunning QBC experiments...")
all_qbc_acc = []
all_qbc_f1 = []
all_qbc_n = []

for run in range(n_runs):
    n_list, acc_list, f1_list = run_active_learning(
        X, y, X_test, y_test, strategy="qbc", k=k, seed=run
    )
    all_qbc_acc.append(acc_list)
    all_qbc_f1.append(f1_list)
    all_qbc_n.append(n_list)
    print(f"  QBC run {run+1}/{n_runs} done ({len(acc_list)} iterations)")

print("\nRunning Random baseline experiments...")
all_rnd_acc = []
all_rnd_f1 = []
all_rnd_n = []

for run in range(n_runs):
    n_list, acc_list, f1_list = run_active_learning(
        X, y, X_test, y_test, strategy="random", k=k, seed=run
    )
    all_rnd_acc.append(acc_list)
    all_rnd_f1.append(f1_list)
    all_rnd_n.append(n_list)
    print(f"  Random run {run+1}/{n_runs} done ({len(acc_list)} iterations)")

# ============================================================
# 6. Aggregate Results
# ============================================================

# All runs have the same number of iterations (same seeds, same stopping criterion)
# Use the n_labeled from the first run as x-axis
n_labeled = all_qbc_n[0]

# Trim all runs to same length (in case of minor differences)
min_len = min(min(len(a) for a in all_qbc_acc), min(len(a) for a in all_rnd_acc))
n_labeled = n_labeled[:min_len]

qbc_acc_matrix = np.array([a[:min_len] for a in all_qbc_acc])
rnd_acc_matrix = np.array([a[:min_len] for a in all_rnd_acc])
qbc_f1_matrix = np.array([a[:min_len] for a in all_qbc_f1])
rnd_f1_matrix = np.array([a[:min_len] for a in all_rnd_f1])

# Mean and standard deviation across runs
qbc_acc_mean = np.mean(qbc_acc_matrix, axis=0)
qbc_acc_std = np.std(qbc_acc_matrix, axis=0)
rnd_acc_mean = np.mean(rnd_acc_matrix, axis=0)
rnd_acc_std = np.std(rnd_acc_matrix, axis=0)

qbc_f1_mean = np.mean(qbc_f1_matrix, axis=0)
qbc_f1_std = np.std(qbc_f1_matrix, axis=0)
rnd_f1_mean = np.mean(rnd_f1_matrix, axis=0)
rnd_f1_std = np.std(rnd_f1_matrix, axis=0)

# ============================================================
# 7. Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Accuracy plot ---
ax = axes[0]
ax.plot(n_labeled, qbc_acc_mean, label="QBC", color="blue", linewidth=2)
ax.fill_between(n_labeled, qbc_acc_mean - qbc_acc_std, qbc_acc_mean + qbc_acc_std,
                color="blue", alpha=0.15)
ax.plot(n_labeled, rnd_acc_mean, label="Random", color="red", linewidth=2)
ax.fill_between(n_labeled, rnd_acc_mean - rnd_acc_std, rnd_acc_mean + rnd_acc_std,
                color="red", alpha=0.15)
ax.set_xlabel("Number of Labeled Samples")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs. Labeled Samples")
ax.legend()
ax.grid(True, alpha=0.3)

# --- F1-score plot ---
ax = axes[1]
ax.plot(n_labeled, qbc_f1_mean, label="QBC", color="blue", linewidth=2)
ax.fill_between(n_labeled, qbc_f1_mean - qbc_f1_std, qbc_f1_mean + qbc_f1_std,
                color="blue", alpha=0.15)
ax.plot(n_labeled, rnd_f1_mean, label="Random", color="red", linewidth=2)
ax.fill_between(n_labeled, rnd_f1_mean - rnd_f1_std, rnd_f1_mean + rnd_f1_std,
                color="red", alpha=0.15)
ax.set_xlabel("Number of Labeled Samples")
ax.set_ylabel("F1-Score")
ax.set_title("F1-Score vs. Labeled Samples")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()
print("\nPlot saved to learning_curves.png")

# ============================================================
# 8. Final Report
# ============================================================

print("\n" + "="*60)
print("FINAL RESULTS (at last iteration)")
print("="*60)
print(f"{'Metric':<15} {'QBC':>15} {'Random':>15}")
print("-"*45)
print(f"{'Accuracy':<15} {qbc_acc_mean[-1]:>15.4f} {rnd_acc_mean[-1]:>15.4f}")
print(f"{'F1-Score':<15} {qbc_f1_mean[-1]:>15.4f} {rnd_f1_mean[-1]:>15.4f}")
print(f"{'Labeled used':<15} {n_labeled[-1]:>15d} {n_labeled[-1]:>15d}")
