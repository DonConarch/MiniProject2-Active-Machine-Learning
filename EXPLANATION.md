# How Our Code Works - For The Report Guys

Hey boys. Here's what the code does. I wrote it so you can write the report without having to actually understand Python. You're welcome.

---

## The Big Picture (What Are We Even Doing?)

Imagine you have 4,601 emails. Some are spam, some are not. You COULD label all of them by hand (boring), OR you could be smart about it and only label the ones that matter most.

That's **Active Learning**. Instead of labeling randomly, we pick the emails that our models are most confused about, label those, and learn faster.

Our specific flavor is called **Query By Committee (QBC)** — basically, we have 3 models that vote on each email, and when they disagree, we say "hey, this email is interesting, let's label it."

Think of it like asking 3 friends if a movie is good. If all 3 say "yes" — boring, you already know. If 2 say "yes" and 1 says "no" — NOW it's interesting, you should actually watch it to find out.

---

## Step by Step (The Code in Human Language)

### 1. Data Handling (lines 12-30)

We load the spambase dataset. It has:
- **4,601 emails**
- **57 features** per email (stuff like "how often does the word FREE appear", "how many CAPITAL LETTERS are there", etc.)
- **1 label**: spam (1) or not spam (0)

We **standardize** the features. This just means we scale everything so no single feature dominates. Like converting all currencies to dollars before comparing prices.

Then we split the data into 3 buckets:
| Bucket | Size | What it's for |
|--------|------|---------------|
| **L** (Labeled) | 10% (~460 emails) | The ones we start with. Our models train on these. |
| **U** (Unlabeled pool) | 80% (~3,680 emails) | The big pile of emails we haven't labeled yet. We pick from here. |
| **T** (Test set) | 10% (~461 emails) | We NEVER touch these during training. Only used at the end to see how good we are. Like a final exam you can't cheat on. |

### 2. The Committee (lines 34-42)

Our "committee" is 3 different models:

1. **Logistic Regression** — draws a straight line (well, a hyperplane) between spam and not-spam. Simple, reliable, like the boring friend who's always kinda right.

2. **Decision Tree** — plays 20 questions. "Does it contain the word FREE? Yes? Does it have lots of exclamation marks? Yes? SPAM." Like that friend who overthinks everything.

3. **Naive Bayes** — uses probability. "Emails with the word MONEY are 80% likely to be spam." Like the friend who just goes with gut feeling and statistics.

They're all different on purpose. If they were all the same, they'd always agree and disagreement would be useless.

### 3. Disagreement (lines 48-69)

For each unlabeled email, all 3 models vote: spam or not spam?

- **All 3 agree** -> disagreement = 0 (boring email, we already know the answer)
- **2 vs 1 split** -> disagreement = 0.33 (interesting! the models are confused!)

Formula: `disagreement = 1 - (votes for winning side / 3)`

### 4. The Active Learning Loop (lines 75-156)

This is the main event. Here's what happens each iteration:

```
REPEAT until 30% of data is labeled:
    1. Train all 3 models on the labeled set L
    2. All 3 models vote on every email in the unlabeled pool U
    3. Find the 10 emails with the MOST disagreement
    4. "Label" those 10 emails (we already know the answers, we just pretend we're asking a human)
    5. Move those 10 from U to L
    6. Check how good we are on the test set
    7. Go back to step 1
```

Each round, we label 10 more emails (k=10), and the models get a little bit smarter.

We also run a **random baseline** — same thing but instead of picking the most confusing emails, we just pick 10 random ones. This lets us prove that QBC is actually smarter than random guessing.

### 5. Why We Run It 20 Times (lines 162-196)

Machine learning has randomness in it (the initial split, etc.). If we ran it once, we might just get lucky (or unlucky).

So we run the whole thing **20 times** with different random seeds and **average the results**. This way our curves are smooth and our conclusions are solid. It's like taking a poll instead of asking one person.

### 6. The Plots (lines 230-263)

We make two learning curves:

**Left plot: Accuracy vs. Number of Labeled Samples**
**Right plot: F1-Score vs. Number of Labeled Samples**

Each plot has:
- **Blue line** = QBC (the smart picking)
- **Red line** = Random (the dumb picking)
- **Shaded area** = standard deviation (how much it varies across our 20 runs)

**What you should see**: The blue line (QBC) should go up FASTER than the red line. This means QBC reaches good performance with FEWER labeled samples. That's the whole point.

### 7. The Metrics (lines 269-276)

- **Accuracy** = what percentage of emails did we classify correctly? Simple.
- **F1-Score** = a fancier metric that cares about both false positives (marking a real email as spam) AND false negatives (letting spam through). Important because the dataset is slightly imbalanced (more not-spam than spam).

---

## What To Write In The Report

Here's basically your cheat sheet:

1. **Introduction**: "We use QBC active learning to classify spam emails. The idea is to label the most informative samples first."

2. **Method**: Explain the 3 models, the vote disagreement, and the loop. Steal from the descriptions above. I won't tell anyone.

3. **Results**: Describe the plots. "QBC reaches X% accuracy with only Y labeled samples, while random sampling needs Z samples to reach the same level."

4. **Conclusion**: "QBC is smarter than random sampling because it focuses on the emails that the models are most uncertain about. This reduces the labeling effort needed to achieve good performance."

---

## How To Run The Code

```bash
cd MiniProject2-Active-Machine-Learning
uv run python main.py
```

Then wait a bit. It prints progress as it goes. At the end you get a plot and final numbers.

---

Good luck with the report. Don't mess it up. :)
