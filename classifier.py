# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Step 2: Load the dataset
# IMPORTANT: Update the file path to where you saved the dataset.
try:
    data = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please download the dataset and place it in the same directory as this script.")
    exit()
except UnicodeDecodeError:
    # Some datasets might require a different encoding
    print("Warning: Failed to read with 'latin-1' encoding. Trying with 'utf-8'.")
    try:
        data = pd.read_csv('spam.csv', encoding='utf-8')
    except Exception as e:
        print(f"Error: Could not read 'spam.csv'. Please check the file format and encoding. Details: {e}")
        exit()

# Step 3: Data Preprocessing
# Rename the columns for clarity. The original columns are 'v1' and 'v2'.
data = data.rename(columns={'v1': 'label', 'v2': 'text'})

# Convert the categorical labels ('ham' and 'spam') into numerical values (0 and 1).
# 'ham' (not spam) will be mapped to 0, and 'spam' will be mapped to 1.
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
print("Data head:")
print(data.head())
print("\nData shape:", data.shape)

# Step 4: Feature Extraction - Vectorizing the text data
# We need to convert the raw text into numerical features that our model can understand.
# TF-IDF (Term Frequency-Inverse Document Frequency) is a great method for this.
# It assigns a weight to each word based on its importance in the document and the corpus.
# We will define this within a pipeline for hyperparameter tuning.

# Split the data into a training set and a testing set.
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# Step 5: Model Improvement with Hyperparameter Tuning
# We use a Pipeline to combine the vectorizer and the model, ensuring the same
# preprocessing steps are applied consistently.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Define the grid of hyperparameters to search through.
# These parameters are for the TfidfVectorizer and LogisticRegression models.
param_grid = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or unigrams and bigrams
    'clf__C': (0.1, 1, 10, 100),            # Regularization parameter
}

# Use GridSearchCV to find the best combination of parameters.
print("\nPerforming GridSearchCV for hyperparameter tuning. This may take a few moments...")
grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=1, cv=5)
grid_search.fit(X_train, y_train)

# The best estimator is the model with the best parameters.
best_model = grid_search.best_estimator_

print("\n--- Hyperparameter Tuning Results ---")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print("Best parameters found:")
print(grid_search.best_params_)
print("-------------------------------------")

# Step 6: Evaluate the best model
# Make predictions on the test set using the best model.
y_pred = best_model.predict(X_test)

# Calculate and print evaluation metrics.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Final Model Evaluation ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("------------------------------")

# Step 7: Make predictions on new, unseen data
def predict_spam(email_text):
    """
    Takes a string of email text, and predicts if it's spam using the best model.
    """
    # The best_model pipeline handles both vectorization and prediction.
    prediction = best_model.predict([email_text])
    probability = best_model.predict_proba([email_text])[0]

    # Map the prediction back to a human-readable label
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    spam_prob = probability[1]

    print(f"\n--- Prediction for new email ---")
    print(f"Email text: '{email_text}'")
    print(f"Predicted class: {result}")
    print(f"Probability of being Spam: {spam_prob:.4f}")
    print("--------------------------------")

# Example usage:
# A typical "ham" message
predict_spam("Hey, are you free for a coffee this evening? Let me know.")

# A typical "spam" message
predict_spam("WINNER! You've been selected to receive a FREE vacation. Claim your prize now!")
