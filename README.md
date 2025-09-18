# Spam Email Classifier

## Overview

This project demonstrates the development of a machine learning model to classify emails as either "spam" or "not spam" (also known as "ham"). The model uses Logistic Regression, a powerful and interpretable algorithm for binary classification tasks. The entire process, from data processing to model evaluation, is contained within a single Python script.

## Features

- Data Preprocessing: Handles raw text data and converts categorical labels into a numerical format.

- Feature Engineering: Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to extract meaningful features from the email text.

- Model Training: Trains a Logistic Regression model to learn the patterns that differentiate spam from ham.

- Hyperparameter Tuning: Employs a pipeline and Grid Search with cross-validation to find the optimal hyperparameters for the model, ensuring the best possible performance.

- Performance Evaluation: Provides key metrics like Accuracy, Precision, Recall, and F1-Score to assess the model's effectiveness.

- Real-time Prediction: Includes a function to make predictions on new, unseen email text.

## Technologies Used

- Python: The core programming language for the project.

- pandas: Used for efficient data loading and manipulation.

- scikit-learn: The primary machine learning library for model building, feature extraction, and evaluation.

## Data Analysis & Processing

The project uses the "SMS Spam Collection" dataset. The raw text data is first cleaned and then transformed into a numerical feature matrix using TfidfVectorizer. This process involves converting text into a Bag of Words representation and weighing the importance of each word.

## Model Used

The model at the heart of this project is Logistic Regression, a linear model for binary classification. It is chosen for its simplicity, speed, and interpretability, making it a great choice for this type of problem.

## Model Training

The model is trained on a portion of the dataset (80% for training) and evaluated on the remaining data (20% for testing). To ensure the model's robustness and optimize its performance, a Pipeline combined with GridSearchCV is used to systematically search for the best combination of hyperparameters, such as the ngram_range for the vectorizer and the C parameter for the logistic regression classifier.

## How to Run the Project

1. Clone the repository:

```bash
git clone <https://github.com/sjain2580/Spam email classifier>
cd <repository_name>
```

2. Create and activate a virtual environment (optional but recommended):python -m venv venv

- On Windows:
  
```bash
.\venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Script:

```bash
python predictive_maintenance.py
```

## Contributors

**<https://github.com/sjain2580>**
Feel free to fork this repository, submit issues, or pull requests to improve the project. Suggestions for model enhancement or additional visualizations are welcome!

## Connect with Me

Feel free to reach out if you have any questions or just want to connect!
**[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sjain04/)**
**[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sjain2580)**
**[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:sjain040395@gmail.com)**

---
