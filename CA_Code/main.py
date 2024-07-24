import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import get_input_data, noise_remover
from Config import *

# Function to evaluate a model for a given target column
def evaluate_model_for_target(target_col, df):
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)

    # Clean the data
    df = noise_remover(df)

    # Split multi-label target into list of labels
    df[target_col] = df[target_col].apply(lambda x: x.split(','))

    X = df[Config.INTERACTION_CONTENT]
    y = df[target_col]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Convert labels to binary format
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    # Train a Chained Classifier with RandomForestClassifier
    model = ClassifierChain(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train_vec, y_train_bin)

    # Predict on the test set
    y_pred_bin = model.predict(X_test_vec)

    # Convert binary predictions back to label format
    y_pred = mlb.inverse_transform(y_pred_bin)
    y_test = mlb.inverse_transform(y_test_bin)

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test_bin, y_pred_bin, target_names=mlb.classes_))

    # Summary of predictions
    correct_predictions = sum([set(pred) == set(true) for pred, true in zip(y_pred, y_test)])
    total_predictions = len(y_test)
    accuracy = accuracy_score(y_test_bin, y_pred_bin, normalize=True, sample_weight=None) * 100
    print(f"\nCorrect Predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    # Group accuracies
    group_accuracies = {
        'AppGallery & Games': [accuracy for i, (pred, true) in enumerate(zip(y_pred, y_test)) if any(label in ['AppGallery-Install/Upgrade', 'AppGallery-Use', 'Coupon/Gifts/Points Issue'] for label in true)],
        'In-App Purchase': [accuracy for i, (pred, true) in enumerate(zip(y_pred, y_test)) if any(label in ['Payment', 'Refund', 'Invoice'] for label in true)],
    }
    for group, accuracies in group_accuracies.items():
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies) * 100
            print(f"Average Accuracy for {group} group: {avg_accuracy:.2f}%")

    overall_avg_accuracy = accuracy_score(y_test_bin, y_pred_bin, normalize=True, sample_weight=None) * 100
    print(f"Overall Average Accuracy for all groups: {overall_avg_accuracy:.2f}%\n")

    return classification_report(y_test_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True)

def main():
    df = get_input_data()
    df = noise_remover(df)

    target_columns = ['y2', 'y3', 'y4']
    for target_col in target_columns:
        print(f"Evaluating target: {target_col}")

        # Evaluate model
        try:
            report = evaluate_model_for_target(target_col, df)
        except Exception as e:
            print(f"Error evaluating target {target_col}: {e}")

if __name__ == "__main__":
    main()