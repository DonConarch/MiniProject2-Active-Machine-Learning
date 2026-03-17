Task:
Implement a Query By Committee (QBC) active learning system to classify data using a categorical dataset.

--------------------------------------------------

1. Data Handling
Dataset to use: 
- Load the dataset with categorical features and a binary label
- Find out which features the dataset includes
- Apply One-Hot Encoding to all features
- Encode labels as binary (0/1)

- Split dataset into:
  L (labeled set): 10%
  U (unlabeled pool): 80%
  T (test set): 10%

--------------------------------------------------

2. Committee Models
Use exactly these 3 models:
- Logistic Regression
- Decision Tree Classifier
- Naive Bayes (GaussianNB if needed)

Train all models ONLY on the labeled set (L)

--------------------------------------------------

3. Query By Committee Loop

Repeat for multiple iterations:

1. Train all models on L
2. Predict labels for all samples in U
3. Compute disagreement for each sample:
   - Use vote-based disagreement
   - Count how many models disagree on the label
4. Select top k most disagreed samples (e.g., k = 10)
5. Move selected samples from U to L
   - Reveal their true labels (simulate labeling)

--------------------------------------------------

4. Stopping Criteria
- Run for a fixed number of iterations OR
- Stop when a labeling budget is reached (e.g., 30% labeled data)

--------------------------------------------------

5. Evaluation
- After training:
  - Evaluate models on test set (T)

- Report:
  - Accuracy
  - F1-score

--------------------------------------------------

6. Baseline Comparison
- Implement random sampling:
  - Randomly select k samples from U instead of using disagreement
- Compare:
  - QBC vs Random Sampling performance

--------------------------------------------------

7. Visualization
- Track performance over iterations:
  - Accuracy vs number of labeled samples

- Plot:
  - QBC curve
  - Random baseline curve

--------------------------------------------------

8. Code Structure
Organize code into:
- Data loading & preprocessing
- Model initialization
- Disagreement calculation function
- Active learning loop
- Evaluation
- Visualization


--------------------------------------------------

9. Constraints
- Use Python with: scikit-learn, numpy, pandas, matplotlib
- Keep code simple and readable
- Add comments explaining each step
- Make the code simple so that it is clearly structured and easy for non-expert coders to understand what is going on. Make it to some degree look like it was done by university students.

--------------------------------------------------

Goal:
Demonstrate that Query By Committee selects more informative samples than random sampling and improves performance with fewer labeled examples.