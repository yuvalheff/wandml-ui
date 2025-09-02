import pandas as pd
import numpy as np
import pickle
from typing import Union

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from iris_species_classification.config import ModelConfig


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on configuration."""
        if self.config.model_type == "LinearDiscriminantAnalysis":
            self.model = LinearDiscriminantAnalysis(**self.config.model_params)
        elif self.config.model_type == "LogisticRegression":
            self.model = LogisticRegression(**self.config.model_params)
        elif self.config.model_type == "SVM":
            self.model = SVC(**self.config.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM with probability=False, use decision_function and convert to probabilities
            decision_scores = self.model.decision_function(X)
            if len(decision_scores.shape) == 1:
                # Binary classification
                exp_scores = np.exp(decision_scores)
                probabilities = exp_scores / (1 + exp_scores)
                return np.column_stack([1 - probabilities, probabilities])
            else:
                # Multi-class classification - use softmax
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            raise ValueError("Model does not support probability prediction.")

    def get_feature_importance(self) -> Union[np.ndarray, None]:
        """
        Get feature importance or coefficients from the model.
        
        Returns:
        np.ndarray or None: Feature importance or coefficients if available.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if hasattr(self.model, 'coef_'):
            return self.model.coef_
        elif hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None

    def save(self, path: str) -> None:
        """
        Save the model as an artifact.

        Parameters:
        path (str): The file path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'ModelWrapper':
        """
        Load the model from a saved artifact.

        Parameters:
        path (str): The file path to load the model from.

        Returns:
        ModelWrapper: The loaded model wrapper.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)