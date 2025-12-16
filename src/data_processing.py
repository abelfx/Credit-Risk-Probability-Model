import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from datetime import datetime
import numpy as np
from xverse.transformer import WOE

def create_proxy_target_variable(df):
    """
    Creates a proxy target variable for credit risk using RFM and K-Means.
    """
    # Convert 'TransactionStartTime' to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Define a snapshot date
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).reset_index()

    # Rename columns
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

    # Pre-process RFM features
    rfm_scaled = StandardScaler().fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Cluster customers using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify the high-risk cluster
    # The cluster with the highest recency, lowest frequency, and lowest monetary value is considered high-risk.
    high_risk_cluster = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().idxmax()['Recency']
    
    # Create the target variable
    rfm['is_high_risk'] = np.where(rfm['Cluster'] == high_risk_cluster, 1, 0)
    
    # Merge the target variable back into the main dataframe
    df = pd.merge(df, rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    
    return df

def build_feature_engineering_pipeline():
    """
    Builds a scikit-learn pipeline for feature engineering with WoE.
    """
    # Define numerical features for scaling
    numerical_features = ['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']

    # Define categorical features for WoE transformation
    categorical_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']

    # Create transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Using WOE transformer for categorical features
    # Note: The WOE transformer from xverse is not a standard sklearn transformer.
    # It doesn't fit neatly into a pipeline that processes X only.
    # It needs to be applied separately or with a custom wrapper.
    # For this implementation, we will apply it in a slightly different manner in the main block.
    
    # For pipeline integration, we would typically wrap it. However, to keep it clear,
    # we will create a preprocessor for numerical and pass-through the rest.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def create_aggregate_features(df):
    """
    Creates aggregate features for each customer.
    """
    agg_features = df.groupby('CustomerId').agg(
        TotalTransactionAmount=('Amount', 'sum'),
        AverageTransactionAmount=('Amount', 'mean'),
        TransactionCount=('Amount', 'count'),
        StdTransactionAmount=('Amount', 'std')
    ).reset_index()
    
    df = pd.merge(df, agg_features, on='CustomerId', how='left')
    return df

if __name__ == '__main__':
    # Load the raw data
    df = pd.read_csv('data/raw/data.csv')

    # Create proxy target variable
    df = create_proxy_target_variable(df)
    
    # Create aggregate features
    df = create_aggregate_features(df)
    
    # Extract Date Features
    df['TransactionHour'] = pd.to_datetime(df['TransactionStartTime']).dt.hour
    df['TransactionDay'] = pd.to_datetime(df['TransactionStartTime']).dt.day
    df['TransactionMonth'] = pd.to_datetime(df['TransactionStartTime']).dt.month
    df['TransactionYear'] = pd.to_datetime(df['TransactionStartTime']).dt.year
    df = df.drop(columns=['TransactionStartTime'])

    # Define features X and target y
    y = df['is_high_risk']
    X = df.drop('is_high_risk', axis=1)

    # Define feature types
    numerical_features = ['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    categorical_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']

    # Apply WoE transformation to categorical features
    woe = WOE()
    X_woe = woe.fit_transform(X[categorical_features], y)
    
    # The rest of the features
    X_other = X.drop(columns=categorical_features)

    # Combine WoE transformed features with the rest
    X_combined = pd.concat([X_woe, X_other], axis=1)

    # Now apply scaling to numerical features
    # We need to make sure we only scale the original numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Create a column transformer for the numerical part
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='passthrough'
    )

    # Fit and transform the combined data
    processed_data = preprocessor.fit_transform(X_combined)

    # Reconstruct the DataFrame
    # Get the columns in the correct order
    remainder_cols = [col for col in X_combined.columns if col not in numerical_features]
    all_feature_names = numerical_features + remainder_cols
    
    processed_df = pd.DataFrame(processed_data, columns=all_feature_names)
    
    # Add the target variable back for saving
    processed_df['is_high_risk'] = y.values
    
    # Save the processed data
    processed_df.to_csv('data/processed/processed_data.csv', index=False)
    
    print("Processed data with WoE transformation saved to data/processed/processed_data.csv")
    print("Shape of processed data:", processed_df.shape)
    print("Columns in processed data:", processed_df.columns.tolist())
