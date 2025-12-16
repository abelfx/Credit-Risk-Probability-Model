import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from datetime import datetime
import numpy as np

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
    Builds a scikit-learn pipeline for feature engineering.
    """
    # Define numerical features for scaling
    numerical_features = ['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount']

    # Define categorical features for one-hot encoding
    categorical_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']

    # Define date features to be extracted
    date_features = ['TransactionStartTime']

    # Create transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a transformer for date features
    class DateTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            X_transformed = X.copy()
            X_transformed['TransactionHour'] = pd.to_datetime(X_transformed['TransactionStartTime']).dt.hour
            X_transformed['TransactionDay'] = pd.to_datetime(X_transformed['TransactionStartTime']).dt.day
            X_transformed['TransactionMonth'] = pd.to_datetime(X_transformed['TransactionStartTime']).dt.month
            X_transformed['TransactionYear'] = pd.to_datetime(X_transformed['TransactionStartTime']).dt.year
            return X_transformed.drop(columns=['TransactionStartTime'])

    date_transformer = DateTransformer()

    # Create a preprocessor to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', date_transformer, date_features)
        ],
        remainder='passthrough'
    )

    # Create the full feature engineering pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    return pipeline

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
    
    # --- Adjusted Script ---
    
    df['TransactionHour'] = pd.to_datetime(df['TransactionStartTime']).dt.hour
    df['TransactionDay'] = pd.to_datetime(df['TransactionStartTime']).dt.day
    df['TransactionMonth'] = pd.to_datetime(df['TransactionStartTime']).dt.month
    df['TransactionYear'] = pd.to_datetime(df['TransactionStartTime']).dt.year
    df = df.drop(columns=['TransactionStartTime'])

    numerical_features = ['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 'StdTransactionAmount', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    categorical_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    processed_data = preprocessor.fit_transform(df)
    
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    # Identify remainder columns that are not transformed
    transformed_cols = numerical_features + categorical_features
    remainder_cols = [col for col in df.columns if col not in transformed_cols]

    # Combine all feature names in the correct order
    all_feature_names = numerical_features + list(cat_feature_names) + remainder_cols
    
    processed_df = pd.DataFrame(processed_data, columns=all_feature_names)
    
    # Save the processed data
    processed_df.to_csv('data/processed/processed_data.csv', index=False)
    
    print("Processed data saved to data/processed/processed_data.csv")
    print("Shape of processed data:", processed_df.shape)
    print("Columns in processed data:", processed_df.columns.tolist())
