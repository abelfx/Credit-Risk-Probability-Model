import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

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
    
    # Create aggregate features
    df = create_aggregate_features(df)
    
    # Build the feature engineering pipeline
    fe_pipeline = build_feature_engineering_pipeline()
    
    # Apply the pipeline to the data
    processed_data = fe_pipeline.fit_transform(df)
    
    # The output of the pipeline is a numpy array. To save it as a CSV with column names, 
    # we need to get the feature names from the preprocessor.
    
    # Get feature names after one-hot encoding
    cat_feature_names = fe_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    # Get remaining feature names
    # The remainder columns are passed through. Let's find out what they are.
    # The columns transformed are numerical_features, categorical_features, and date_features.
    # We need to be careful about the order.
    
    # Let's create a new dataframe with the processed data
    
    # For simplicity for now, let's just save the numpy array
    # A more robust solution would be to reconstruct the DataFrame with proper column names.
    
    # For now, let's just demonstrate the pipeline creation and transformation.
    # In a real scenario, you would save this processed_data to be used by the training script.
    
    print("Feature engineering pipeline created and applied.")
    print("Shape of processed data:", processed_data.shape)

    # To properly save the data, we need to handle the column names which can be complex with ColumnTransformer.
    # Here is a way to do it:
    
    # Get the numerical feature names (they remain the same)
    num_features = numerical_features
    
    # Get the date feature names
    date_feature_names = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']

    # Get the list of columns that are passed through
    remainder_cols = [col for col in df.columns if col not in numerical_features + categorical_features + date_features]

    # Combine all feature names
    all_feature_names = num_features + list(cat_feature_names) + date_feature_names + remainder_cols
    
    # It seems there is an issue with the column ordering and what is passed as remainder.
    # A simpler approach for the script is to do the date transformation first, then apply the preprocessor.

    # Let's adjust the script for clarity and correctness of saving.
    
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
