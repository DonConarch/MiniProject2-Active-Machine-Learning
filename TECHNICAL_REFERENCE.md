# Technical Reference - QBC Active Learning on Spambase

Everything you need for the report, with citations, formulas, and exact parameter values.

---

## 1. Dataset: Spambase (UCI Machine Learning Repository)

**Source**: UCI Machine Learning Repository — Spambase Data Set
**Citation**: Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase Data Set. UCI Machine Learning Repository.

| Property | Value |
|---|---|
| Total samples | 4,601 |
| Features | 57 (all continuous) |
| Classes | 2 (spam = 1, not spam = 0) |
| Class distribution | 1,813 spam (39.4%), 2,788 not spam (60.6%) |
| Missing values | None |

### Feature breakdown (57 features)

| Feature group | Count | Description | Range |
|---|---|---|---|
| `word_freq_*` | 48 | Percentage of words in the email matching a specific word (e.g., "free", "money", "credit") | [0, 100] |
| `char_freq_*` | 6 | Percentage of characters matching a specific character (`;`, `(`, `[`, `!`, `$`, `#`) | [0, 100] |
| `capital_run_length_average` | 1 | Average length of consecutive capital letters | [1, ...] |
| `capital_run_length_longest` | 1 | Longest run of consecutive capital letters | [1, ...] |
| `capital_run_length_total` | 1 | Total number of capital letters in the email | [1, ...] |

### Preprocessing applied

- **StandardScaler** (z-score normalization): Each feature transformed to zero mean and unit variance.
  - Formula: `z = (x - mu) / sigma`
  - Required because Logistic Regression is sensitive to feature scale. Decision Tree and Naive Bayes are scale-invariant but are not harmed by it.
- **No one-hot encoding** needed — all features are already continuous (unlike the mushroom dataset which has categorical features).

---

## 2. Data Split

| Set | Symbol | Fraction | Approximate size | Purpose |
|---|---|---|---|---|
| Labeled | L | 10% | ~460 | Initial training data for the committee |
| Unlabeled pool | U | 80% | ~3,680 | Candidates for querying; labels hidden |
| Test | T | 10% | ~461 | Held-out evaluation; never used during training |

- Split method: `sklearn.model_selection.train_test_split`
- **Stratified** splitting used (preserves class ratio in all subsets)
- Test set is **fixed** across all runs (`random_state=42`) to ensure fair comparison
- L and U are re-split with different seeds per run to introduce variability

---

## 3. Committee Models

We use a heterogeneous committee of 3 classifiers. Diversity is achieved through **algorithmic diversity** — different model families with different inductive biases.

### 3.1 Logistic Regression

| Parameter | Value | Justification |
|---|---|---|
| `max_iter` | 1000 | Ensures convergence on the 57-dimensional feature space |
| `random_state` | 42 | Reproducibility |
| Solver | `lbfgs` (default) | Efficient for small-to-medium datasets |
| Regularization | L2 (default, `C=1.0`) | Prevents overfitting with few labeled samples |

**How it works**: Models the log-odds of the positive class as a linear function of features:

```
log(P(spam) / P(not spam)) = w^T x + b
```

Fits weights `w` via maximum likelihood. Decision boundary is a hyperplane in feature space.

**Strengths for QBC**: Provides smooth probability estimates, linear decision boundary complements the non-linear models.

### 3.2 Decision Tree Classifier

| Parameter | Value | Justification |
|---|---|---|
| `random_state` | 42 | Reproducibility |
| All others | sklearn defaults | `criterion='gini'`, no max depth, `min_samples_split=2` |

**How it works**: Recursively partitions the feature space using axis-aligned splits. At each node, selects the feature and threshold that maximizes information gain (Gini impurity reduction):

```
Gini(S) = 1 - sum_c(p_c^2)
```

where `p_c` is the fraction of samples belonging to class `c`.

**Strengths for QBC**: Highly non-linear, creates complex decision boundaries. Prone to overfitting on small labeled sets, which actually *increases* disagreement early on — helpful for QBC.

### 3.3 Gaussian Naive Bayes

| Parameter | Value | Justification |
|---|---|---|
| All defaults | — | No hyperparameters to tune |

**How it works**: Assumes features are conditionally independent given the class, and each feature follows a Gaussian distribution:

```
P(x_j | class=c) = N(mu_jc, sigma_jc^2)
```

Classification via Bayes' rule:

```
P(class=c | x) proportional to P(class=c) * product_j(P(x_j | class=c))
```

**Strengths for QBC**: Very different inductive bias from the other two models. The conditional independence assumption is "wrong" but produces genuinely different predictions, increasing committee diversity.

### Why these 3 models?

The committee must be **diverse** for QBC to work. If all models agree on everything, disagreement is always zero and QBC degenerates to random sampling. Our committee has:

1. A **linear** model (Logistic Regression)
2. A **non-linear, deterministic** model (Decision Tree)
3. A **probabilistic, generative** model (Naive Bayes)

These have fundamentally different decision boundaries, ensuring meaningful disagreement.

---

## 4. Query Strategy: Vote-Based Disagreement

### Definition

For a committee of `C` models and a sample `x`:

1. Each model `m_c` predicts a label: `y_hat_c = m_c(x)`
2. Count votes for the most popular label: `v_max = max(count(y_hat = 0), count(y_hat = 1))`
3. Disagreement: `D(x) = 1 - v_max / C`

### Possible values (with C = 3 models, binary classification)

| Vote pattern | v_max | D(x) | Interpretation |
|---|---|---|---|
| 3-0 (unanimous) | 3 | 0.000 | Full agreement — sample is "easy" |
| 2-1 (split) | 2 | 0.333 | One model disagrees — sample is "informative" |

### Selection step

At each iteration, compute `D(x)` for all `x` in the unlabeled pool `U`, then select the **top-k** samples with highest disagreement. These are moved from `U` to `L` with their true labels revealed.

### Why vote disagreement?

Alternative disagreement measures exist (vote entropy, KL-divergence). We chose vote disagreement because:
- Simple to implement and interpret
- Requires only hard predictions (no probability calibration needed)
- With 3 models and 2 classes, it reduces to a binary signal (agree vs. disagree), which is clean and interpretable

---

## 5. Active Learning Loop — Pseudocode

```
Input: X, y, k=10, max_fraction=0.30
Split into L (10%), U (80%), T (10%)

WHILE |L| <= max_fraction * |X| AND |U| > 0:
    # Train
    FOR each model m in committee:
        m.fit(L)

    # Evaluate
    predictions = [m.predict(T) for m in committee]
    majority_vote = mode(predictions)
    record accuracy(T, majority_vote) and F1(T, majority_vote)

    # Query
    IF strategy == "QBC":
        D = compute_disagreement(committee, U)
        query_indices = argsort(D)[-k:]     # top-k most disagreed
    ELSE:  # random baseline
        query_indices = random_sample(k, from=U)

    # Update
    Move U[query_indices] -> L (reveal true labels)

RETURN performance_history
```

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| k (query batch size) | 10 | 10 new labels per iteration |
| max_fraction | 0.30 | Stop at 30% of total data labeled (~1,380 samples) |
| n_runs | 20 | Number of independent runs for averaging |
| Initial L size | ~460 | 10% of 4,601 |
| Iterations per run | ~92 | (1380 - 460) / 10 = 92 iterations |

---

## 6. Evaluation

### Test set evaluation

The committee's predictions are combined via **majority vote**:

```
y_hat(x) = mode(m_1(x), m_2(x), m_3(x))
```

Implementation: since predictions are 0/1 and we have 3 models, `mean >= 0.5` is equivalent to majority vote.

### Metrics

**Accuracy**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**F1-Score** (harmonic mean of precision and recall):
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

F1 is important here because the dataset is **imbalanced** (60.6% not-spam vs 39.4% spam). A naive classifier predicting "not spam" always would get 60.6% accuracy but F1 = 0 for the spam class.

### Averaging over runs

Each experiment is repeated 20 times with different random seeds. We report:
- **Mean** performance curve across runs
- **Standard deviation** shown as a shaded band around the mean

This reduces the effect of any single lucky/unlucky split and gives statistically meaningful results.

---

## 7. Baseline: Random Sampling

Identical to QBC except the query step selects `k` samples **uniformly at random** from `U` instead of by disagreement. Same initial split, same models, same evaluation.

**Purpose**: Demonstrates the *value of active learning*. If QBC's learning curve rises faster than random's, it proves that selecting by disagreement is more label-efficient.

**Expected result**: QBC should reach the same accuracy as random sampling but with fewer labeled samples. The gap between the curves quantifies the label savings.

---

## 8. Visualization Details

Two side-by-side learning curves:

| Plot | X-axis | Y-axis | Lines |
|---|---|---|---|
| Left | Number of labeled samples | Accuracy | QBC (blue), Random (red) |
| Right | Number of labeled samples | F1-Score | QBC (blue), Random (red) |

- Solid lines = mean over 20 runs
- Shaded bands = +/- 1 standard deviation
- Grid enabled for readability
- Saved as `learning_curves.png` at 150 DPI

### What to look for in the plots

1. **Does QBC rise faster?** If the blue line is above the red line in the early iterations (few labels), QBC is more label-efficient.
2. **Do they converge?** With enough labels, both strategies should approach similar accuracy — there's a ceiling determined by the models and features.
3. **How tight are the bands?** Narrow bands mean consistent performance; wide bands suggest sensitivity to the initial split.
4. **Where is the biggest gap?** The x-value where QBC most outperforms random is the "sweet spot" for active learning.

---

## 9. Key Concepts for the Report

### Active Learning
A semi-supervised machine learning paradigm where the learner can interactively query an oracle (e.g., a human annotator) to label data points. Goal: achieve high performance with minimal labeling effort.

### Query By Committee (QBC)
An active learning strategy (Seung et al., 1992) where a committee of models is trained on the current labeled set, and the sample with the highest committee disagreement is selected for labeling. Theoretical basis: selecting samples that maximally reduce the version space.

**Citation**: Seung, H. S., Opper, M., & Sompolinsky, H. (1992). Query by committee. *Proceedings of the Fifth Annual Workshop on Computational Learning Theory*, 287-294.

### Pool-Based Active Learning
Our setting: a large pool of unlabeled data exists upfront, and the learner selects from this pool. Contrast with stream-based (samples arrive one at a time) and membership query synthesis (learner generates new samples).

### Label Efficiency
The core claim of active learning: by choosing which samples to label, we can achieve the same performance as random sampling while using **fewer labels**. This is measured by comparing the area under the learning curves.

---

## 10. Reproducibility

| Aspect | How ensured |
|---|---|
| Random splits | `random_state` parameter in `train_test_split` |
| Model initialization | `random_state=42` for LR and DT |
| Random baseline | `np.random.RandomState(seed)` per run |
| Test set | Fixed across all runs with `random_state=42` |
| Library versions | Pinned in `pyproject.toml` (scikit-learn >= 1.6.0, numpy >= 2.4.3) |

---

## 11. Limitations (Good to mention in report discussion)

1. **Committee size is small (3)** — disagreement can only be 0 or 0.33. A larger committee would give finer-grained disagreement scores.
2. **Vote disagreement is coarse** — vote entropy or probability-based measures (KL divergence) would capture uncertainty more precisely.
3. **No model tuning** — all models use default hyperparameters. Performance could improve with cross-validation.
4. **Simulated oracle** — we reveal true labels from the dataset. Real active learning involves a human annotator who may introduce noise.
5. **Single dataset** — results may not generalize. Testing on additional datasets would strengthen conclusions.
6. **Cold start** — with only 10% initial labels, the Decision Tree may overfit, producing noisy early disagreements.

---

## 12. Possible Extensions (to mention as future work)

- **Vote entropy** as an alternative disagreement measure: `H = -sum_y (V(y)/C) * log(V(y)/C)` where `V(y)` is the number of votes for class `y`
- **Bagging-based QBC** — train the same model on bootstrap samples of L, creating diversity through data perturbation rather than model type
- **Cost-sensitive evaluation** — misclassifying spam as legitimate (false negative) vs. marking legitimate email as spam (false positive) may have different costs
- **Comparison with uncertainty sampling** — use a single model's predicted probability to select uncertain samples
