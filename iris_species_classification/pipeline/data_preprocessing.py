from typing import Optional
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from iris_species_classification.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config
        self.label_encoder = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable.

        Returns:
        DataProcessor: The fitted processor.
        """
        if y is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        # Drop Id column if it exists
        X_transformed = X.copy()
        if 'Id' in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=['Id'])
            
        # Select only the feature columns we need
        feature_columns = self.config.feature_columns
        X_transformed = X_transformed[feature_columns]
        
        return X_transformed
    
    def transform_target(self, y: pd.Series) -> pd.Series:
        """
        Transform the target variable using the label encoder.
        
        Parameters:
        y (pd.Series): The target variable.
        
        Returns:
        pd.Series: The encoded target variable.
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call fit() first.")
        return pd.Series(self.label_encoder.transform(y), index=y.index, name=y.name)
    
    def inverse_transform_target(self, y_encoded):
        """
        Inverse transform the encoded target variable.
        
        Parameters:
        y_encoded: The encoded target variable.
        
        Returns:
        The original target labels.
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call fit() first.")
        return self.label_encoder.inverse_transform(y_encoded)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable.

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
