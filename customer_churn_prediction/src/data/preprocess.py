import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_raw_data(path: str) -> pd.DataFrame:

    # Load customer churn data from CSV.

    df = pd.read_csv(path)

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Drop non-predictive identifier
    df.drop(columns=['customerID'], inplace=True)

    # Fix TotalCharges data type
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Business - aware imputation
    df.fillna({'TotalCharges': 0}, inplace=True)

    # Encode Target variable
    df['Churn']  = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def split_features_target(df: pd.DataFrame):

    # separate features and target

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y

def get_feature_types(X: pd.DataFrame):

    # Identify categorical and numerical features

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    return categorical_features, numerical_features

def build_preprocessor(categorical_features, numerical_features):

    # Build ColumnTransformer for preprocessing

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    return preprocessor

def prepare_data(data_path: str):

    '''
    Full Preprocessing Workflow

    '''

    df = load_raw_data(data_path)

    df = clean_data(df)

    X,y = split_features_target(df)

    categorical_features, numerical_features = get_feature_types(X)

    preprocessor = build_preprocessor(categorical_features, numerical_features)

    return X, y, preprocessor







    





    




    