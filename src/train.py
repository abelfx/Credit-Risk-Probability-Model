import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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

    # Drop non-feature columns
    if 'CustomerId' in df.columns:
        df = df.drop(columns=['CustomerId'])
    
    # Handle potential non-numeric columns that are not features and drop rows with NaN values
    df = df.select_dtypes(include=np.number).dropna()


    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Set MLflow experiment
    mlflow.set_experiment("Credit Risk Model")

    # --- Logistic Regression ---
    with mlflow.start_run(run_name="Logistic Regression"):
        # Initialize and train model
        lr = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter for convergence
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


    # --- Random Forest with Hyperparameter Tuning ---
    with mlflow.start_run(run_name="Random Forest with GridSearch"):
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100], # Reduced for speed
            'max_depth': [10, None],
            'min_samples_split': [2, 5]
        }

        # Initialize GridSearchCV
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1')

        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_rf = grid_search.best_estimator_

        # Make predictions
        y_pred_rf = best_rf.predict(X_test)
        y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

        # Evaluate model
        accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf)

        # Log parameters and metrics
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy_rf)
        mlflow.log_metric("precision", precision_rf)
        mlflow.log_metric("recall", recall_rf)
        mlflow.log_metric("f1_score", f1_rf)
        mlflow.log_metric("roc_auc", roc_auc_rf)

        # Log model
        mlflow.sklearn.log_model(best_rf, "random_forest_model")
        print("Random Forest model with GridSearch trained and logged.")

    # --- Model Registration ---
    # Programmatically select the best model based on F1 score.
    
    # Find the best run
    best_run = mlflow.search_runs(order_by=["metrics.f1_score DESC"]).iloc[0]
    best_run_id = best_run.run_id
    
    # Determine which model it was
    if "Logistic Regression" in best_run.data.tags.get('mlflow.runName', ''):
        model_name_from_run = "logistic_regression_model"
    else:
        model_name_from_run = "random_forest_model"

    model_uri = f"runs:/{best_run_id}/{model_name_from_run}"
    
    print(f"Registering {model_name_from_run} as the best model.")

    # Register the best model
    model_name = "CreditRiskModel"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered under the name: {model_name}")
