from typing import Optional
import pandas as pd
import pickle
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from iris_species_classification.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        self.scaler = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # First, create engineered features if configured
        X_with_features = self._create_engineered_features(X) if self.config.apply_feature_engineering else X
        
        # Then fit the scaler on the complete feature set (including engineered features)
        if self.config.apply_scaling:
            self.scaler = StandardScaler()
            self.scaler.fit(X_with_features)
        return self

    def _create_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on the configuration.
        
        Parameters:
        X (pd.DataFrame): The input features.
        
        Returns:
        pd.DataFrame: Features with engineered features added.
        """
        X_engineered = X.copy()
        
        # Create the 6 engineered features as specified in the experiment plan
        if "SepalRatio" in self.config.engineered_features:
            X_engineered["SepalRatio"] = X["SepalLengthCm"] / X["SepalWidthCm"]
            
        if "PetalRatio" in self.config.engineered_features:
            X_engineered["PetalRatio"] = X["PetalLengthCm"] / X["PetalWidthCm"]
            
        if "SepalArea" in self.config.engineered_features:
            X_engineered["SepalArea"] = X["SepalLengthCm"] * X["SepalWidthCm"]
            
        if "PetalArea" in self.config.engineered_features:
            X_engineered["PetalArea"] = X["PetalLengthCm"] * X["PetalWidthCm"]
            
        if "TotalLength" in self.config.engineered_features:
            X_engineered["TotalLength"] = X["SepalLengthCm"] + X["PetalLengthCm"]
            
        if "TotalWidth" in self.config.engineered_features:
            X_engineered["TotalWidth"] = X["SepalWidthCm"] + X["PetalWidthCm"]
        
        return X_engineered

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features based on the configuration.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        X_transformed = X.copy()
        
        # Create engineered features first if configured
        if self.config.apply_feature_engineering:
            X_transformed = self._create_engineered_features(X_transformed)
        
        # Apply scaling if configured (on all features including engineered ones)
        if self.config.apply_scaling and self.scaler is not None:
            X_scaled = self.scaler.transform(X_transformed)
            X_transformed = pd.DataFrame(
                X_scaled, 
                columns=X_transformed.columns, 
                index=X_transformed.index
            )
        
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input features.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the feature processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'FeatureProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
