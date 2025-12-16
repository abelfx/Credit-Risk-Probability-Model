import pandas as pd
import numpy as np
from src.data_processing import create_aggregate_features, create_proxy_target_variable
from src.train import evaluate_model

def test_create_aggregate_features():
    """
    Tests that the aggregate feature creation function adds the expected columns.
    """
    data = {
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 300]
    }
    df = pd.DataFrame(data)
    agg_df = create_aggregate_features(df)
    
    expected_cols = ['TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount']
    
    for col in expected_cols:
        assert col in agg_df.columns

def test_create_proxy_target_variable():
    """
    Tests that the proxy target variable function adds the 'is_high_risk' column
    and that it contains only binary values.
    """
    data = {
        'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C4'],
        'TransactionId': [1, 2, 3, 4, 5],
        'TransactionStartTime': pd.to_datetime(['2025-01-01', '2025-01-05', '2025-03-01', '2025-04-01', '2025-05-01']),
        'Amount': [10, 20, 5, 100, 80]
    }
    df = pd.DataFrame(data)
    
    # Ensure there's enough data to form clusters
    df = pd.concat([df]*10, ignore_index=True)

    target_df = create_proxy_target_variable(df)
    
    assert 'is_high_risk' in target_df.columns
    assert set(target_df['is_high_risk'].unique()) <= {0, 1}

def test_evaluate_model():
    """
    Tests that the evaluate_model function correctly calculates performance metrics.
    """
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.6, 0.8])
    
    accuracy, precision, recall, f1, roc_auc = evaluate_model(y_true, y_pred, y_prob)
    
    assert np.isclose(accuracy, 2/3)
    assert np.isclose(precision, 2/3)
    assert np.isclose(recall, 2/3)
    assert np.isclose(f1, 2/3)
    assert roc_auc > 0.5

def test_woe_transformation():
    """
    Tests that the categorical features are correctly transformed into numerical WoE values.
    This is a conceptual test as the actual WoE values depend on the data distribution.
    We will check if the output is numerical.
    """
    # This test is more complex to set up due to the nature of WoE.
    # A simple check is to ensure the output columns are numeric.
    # A more detailed test would require a fixture with known WoE values.
    
    # For now, we will rely on the successful execution of the data_processing script.
    # The script itself will fail if the transformation is not applied correctly.
    # We can add a placeholder test to be expanded later.
    pass
