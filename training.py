from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import pandas as pd

# load data
data = pd.read_csv("processed_data.csv")

# Split the data into features (X) and labels (y)
X = data["processed_text"]
y = data["label"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display the shapes of the resulting sets
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Create a pipeline with TF-IDF vectorizer (Term Frequency-Inverse Document Frequency)  and a classifier - Multinomial Naive Bayes
model = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())])

# Train the model on the training data
model.fit(X_train, y_train)

# Predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Multinomial Accuracy:", accuracy)
print("Multinomial Classification Report:\n", report)

# Create a pipeline with TF-IDF vectorizer (Term Frequency-Inverse Document Frequency)  and a classifier - Support Vector Machine
model_svm = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", SVC())])

# Train the SVM model on the training data
model_svm.fit(X_train, y_train)

# Predictions on the testing data
y_pred_svm = model_svm.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print("SVM Accuracy:", accuracy_svm)
print("SVM Classification Report:\n", report_svm)
