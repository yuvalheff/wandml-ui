"""
ML Pipeline for Iris Species Classification

Complete pipeline that combines data processing, feature processing, 
and model prediction for MLflow deployment.
"""

import pandas as pd
import numpy as np
from typing import Union, List

from iris_species_classification.pipeline.data_preprocessing import DataProcessor
from iris_species_classification.pipeline.feature_preprocessing import FeatureProcessor
from iris_species_classification.pipeline.model import ModelWrapper
from iris_species_classification.config import Config


class ModelPipeline:
    """
    Complete ML pipeline for Iris species classification.
    
    This class combines data processing, feature processing, and model prediction
    into a single deployable pipeline suitable for MLflow.
    """
    
    def __init__(self, data_processor: DataProcessor, feature_processor: FeatureProcessor, 
                 model_wrapper: ModelWrapper, class_names: List[str]):
        """
        Initialize the pipeline components.
        
        Parameters:
        data_processor: Fitted data processor
        feature_processor: Fitted feature processor
        model_wrapper: Fitted model wrapper
        class_names: List of class names in order
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model_wrapper = model_wrapper
        self.class_names = class_names
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Parameters:
        X: Input features (raw data with same structure as training data)
        
        Returns:
        np.ndarray: Predicted class labels (encoded)
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            # Assume columns are in the same order as training
            columns = self.data_processor.config.feature_columns
            if 'Id' not in columns:
                columns = ['Id'] + columns  # Add Id column if not present
            X = pd.DataFrame(X, columns=columns)
        
        # Apply data preprocessing (remove Id, select features)
        X_processed = self.data_processor.transform(X)
        
        # Apply feature preprocessing (scaling)
        X_features = self.feature_processor.transform(X_processed)
        
        # Make predictions
        predictions = self.model_wrapper.predict(X_features)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Parameters:
        X: Input features (raw data with same structure as training data)
        
        Returns:
        np.ndarray: Predicted class probabilities
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            # Assume columns are in the same order as training
            columns = self.data_processor.config.feature_columns
            if 'Id' not in columns:
                columns = ['Id'] + columns  # Add Id column if not present
            X = pd.DataFrame(X, columns=columns)
        
        # Apply data preprocessing (remove Id, select features)
        X_processed = self.data_processor.transform(X)
        
        # Apply feature preprocessing (scaling)
        X_features = self.feature_processor.transform(X_processed)
        
        # Make probability predictions
        probabilities = self.model_wrapper.predict_proba(X_features)
        
        return probabilities
    
    def predict_with_names(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """
        Make predictions and return class names instead of encoded labels.
        
        Parameters:
        X: Input features (raw data with same structure as training data)
        
        Returns:
        List[str]: Predicted class names
        """
        predictions = self.predict(X)
        
        # Convert encoded predictions back to class names
        if hasattr(self.data_processor, 'label_encoder') and self.data_processor.label_encoder is not None:
            return self.data_processor.inverse_transform_target(predictions).tolist()
        else:
            # If no label encoder, assume predictions are indices into class_names
            return [self.class_names[pred] for pred in predictions]
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names used by the pipeline.
        
        Returns:
        List[str]: Feature names (including engineered features)
        """
        feature_names = self.data_processor.config.feature_columns.copy()
        
        # Add engineered feature names if feature engineering is enabled
        if (hasattr(self.feature_processor.config, 'apply_feature_engineering') and 
            self.feature_processor.config.apply_feature_engineering):
            feature_names.extend(self.feature_processor.config.engineered_features)
            
        return feature_names
    
    def get_class_names(self) -> List[str]:
        """
        Get the class names used by the pipeline.
        
        Returns:
        List[str]: Class names
        """
        return self.class_names.copy()
