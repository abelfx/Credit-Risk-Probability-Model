import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates and returns evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return accuracy, precision, recall, f1, roc_auc

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/processed/processed_data.csv')

    # Define features and target
    # Drop non-feature columns. This needs to be robust.
    # Assuming 'is_high_risk' is the target and 'CustomerId' is an identifier.
    # The preprocessor might have introduced other non-feature columns.
    # For this example, let's assume all other columns are features.
    
    # A better approach would be to save features list during processing.
    # For now, we will drop known non-feature columns.
    
    # Ensure CustomerId is handled correctly if it's in the processed data
    if 'CustomerId' in df.columns:
        df = df.drop(columns=['CustomerId'])
    
    # Handle potential non-numeric columns that are not features
    # For example, if there are any object type columns left that are not supposed to be there.
    df = df.select_dtypes(include=np.number)


    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Set MLflow experiment
    mlflow.set_experiment("Credit Risk Model")

    # --- Logistic Regression ---
    with mlflow.start_run(run_name="Logistic Regression"):
        # Initialize and train model
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)

        # Make predictions
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]

        # Evaluate model
        accuracy_lr, precision_lr, recall_lr, f1_lr, roc_auc_lr = evaluate_model(y_test, y_pred_lr, y_prob_lr)

        # Log parameters and metrics
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_metric("accuracy", accuracy_lr)
        mlflow.log_metric("precision", precision_lr)
        mlflow.log_metric("recall", recall_lr)
        mlflow.log_metric("f1_score", f1_lr)
        mlflow.log_metric("roc_auc", roc_auc_lr)

        # Log model
        mlflow.sklearn.log_model(lr, "logistic_regression_model")
        print("Logistic Regression model trained and logged.")


    # --- Random Forest ---
    with mlflow.start_run(run_name="Random Forest"):
        # Initialize and train model
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]

        # Evaluate model
        accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf)

        # Log parameters and metrics
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_metric("accuracy", accuracy_rf)
        mlflow.log_metric("precision", precision_rf)
        mlflow.log_metric("recall", recall_rf)
        mlflow.log_metric("f1_score", f1_rf)
        mlflow.log_metric("roc_auc", roc_auc_rf)

        # Log model
        mlflow.sklearn.log_model(rf, "random_forest_model")
        print("Random Forest model trained and logged.")

    # --- Model Registration ---
    # This part should be run after analyzing the results in the MLflow UI.
    # For automation, we can programmatically select the best model.
    
    # Let's assume the best model is the one with the highest F1 score.
    if f1_lr > f1_rf:
        best_run_id = mlflow.search_runs(filter_string="metrics.f1_score = {}".format(f1_lr)).iloc[0].run_id
        model_uri = f"runs:/{best_run_id}/logistic_regression_model"
        print("Registering Logistic Regression model.")
    else:
        best_run_id = mlflow.search_runs(filter_string="metrics.f1_score = {}".format(f1_rf)).iloc[0].run_id
        model_uri = f"runs:/{best_run_id}/random_forest_model"
        print("Registering Random Forest model.")

    # Register the best model
    model_name = "CreditRiskModel"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered under the name: {model_name}")
