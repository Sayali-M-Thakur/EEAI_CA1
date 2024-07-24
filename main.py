import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import get_input_data, noise_remover, translate_to_en
from Config import *


def evaluate_model_for_target(target_col, df):
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)

    # Translate if needed
    # df[Config.INTERACTION_CONTENT] = translate_to_en(df[Config.INTERACTION_CONTENT].tolist())

    # Clean the data
    df = noise_remover(df)

    # Check for missing values and handle them
    if df[Config.INTERACTION_CONTENT].isnull().any() or df[target_col].isnull().any():
        print(
            f"Missing values detected in {Config.INTERACTION_CONTENT} or {target_col}. Dropping rows with missing values.")
        df = df.dropna(subset=[Config.INTERACTION_CONTENT, target_col])

    X = df[Config.INTERACTION_CONTENT]
    y = df[target_col]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_vec)

    # Get unique labels from the test set and predictions
    unique_labels = sorted(set(y_test) | set(y_pred))

    # Evaluate the model
    report = classification_report(y_test, y_pred, labels=unique_labels, target_names=unique_labels)
    accuracy = accuracy_score(y_test, y_pred)

    return report, accuracy, X_test, y_test, y_pred


def print_detailed_results(content, y_test, y_pred):
    results = []
    for text, true, pred in zip(content, y_test, y_pred):
        correct_preds = sum(1 for t, p in zip(true, pred) if t == p)
        accuracy = correct_preds / len(true)
        result = f"""
        Content: {text}

        Predicted Classes: {pred}

        True Classes: {true}

        Accuracy: {accuracy * 100:.2f}%
        """
        results.append(result)
    return results


def main():
    df = get_input_data()
    df = noise_remover(df)

    target_columns = ['y2', 'y3', 'y4']
    overall_accuracies = []

    for target_col in target_columns:
        print(f"Evaluating target: {target_col}")

        # Evaluate model
        try:
            report, accuracy, X_test, y_test, y_pred = evaluate_model_for_target(target_col, df)
            overall_accuracies.append(accuracy)
            print(report)
            print(f"\nCorrect Predictions: {sum(y_test == y_pred)}/{len(y_test)}")
            print(f"Accuracy: {accuracy * 100:.2f}%\n")

            # Detailed per-sample predictions and accuracies
            detailed_results = print_detailed_results(X_test.tolist(), y_test.tolist(), y_pred.tolist())
            for result in detailed_results:
                print(result)

        except Exception as e:
            print(f"Error evaluating target {target_col}: {e}")

    # Overall Average Accuracy
    overall_avg_accuracy = sum(overall_accuracies) / len(overall_accuracies) * 100
    print(f"\nOverall Average Accuracy for all groups: {overall_avg_accuracy:.2f}%")


if __name__ == "__main__":
    main()