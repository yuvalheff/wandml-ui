import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import os
import json

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer

from iris_species_classification.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
        self.app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]

    def evaluate_model(self, model_wrapper, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame, y_test: pd.Series, class_names: List[str],
                      output_dir: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation including metrics and visualizations.
        """
        results = {}
        
        # Predictions
        y_pred = model_wrapper.predict(X_test)
        y_proba = model_wrapper.predict_proba(X_test)
        
        # Basic metrics
        results.update(self._calculate_basic_metrics(y_test, y_pred, y_proba, class_names))
        
        # Cross-validation scores
        results.update(self._calculate_cv_scores(model_wrapper, X_train, y_train))
        
        # Generate visualizations
        self._generate_confusion_matrix_plot(y_test, y_pred, class_names, output_dir)
        self._generate_roc_curves_plot(y_test, y_proba, class_names, output_dir)
        self._generate_feature_importance_plot(model_wrapper, X_train.columns.tolist(), output_dir)
        self._generate_prediction_confidence_plot(y_test, y_pred, y_proba, class_names, output_dir)
        self._generate_class_distribution_plot(y_train, y_test, class_names, output_dir)
        
        return results

    def _calculate_basic_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                y_proba: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """Calculate basic classification metrics."""
        metrics = {}
        
        # Basic scores
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # AUC scores (one-vs-rest)
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_true)
        if y_true_binarized.shape[1] == 1:  # Binary case, expand to 2 columns
            y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
        
        metrics['macro_averaged_auc'] = roc_auc_score(y_true_binarized, y_proba, average='macro', multi_class='ovr')
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics['per_class_metrics'] = class_report
        
        # Individual class AUC scores
        class_auc_scores = {}
        for i, class_name in enumerate(class_names):
            if i < y_proba.shape[1]:
                class_auc_scores[class_name] = roc_auc_score(y_true_binarized[:, i], y_proba[:, i])
        metrics['class_auc_scores'] = class_auc_scores
        
        return metrics

    def _calculate_cv_scores(self, model_wrapper, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Calculate cross-validation scores."""
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=self.config.cv_shuffle,
            random_state=self.config.random_state
        )
        
        cv_scores = {}
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            scores = cross_val_score(model_wrapper.model, X_train, y_train, cv=cv, scoring=metric)
            cv_scores[f'cv_{metric}_mean'] = scores.mean()
            cv_scores[f'cv_{metric}_std'] = scores.std()
        
        return cv_scores

    def _generate_confusion_matrix_plot(self, y_true: pd.Series, y_pred: np.ndarray, 
                                       class_names: List[str], output_dir: str):
        """Generate confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(class_names))),
                ticktext=class_names
            ),
            yaxis=dict(
                tickmode="array", 
                tickvals=list(range(len(class_names))),
                ticktext=class_names
            )
        )
        
        self._apply_color_theme(fig)
        fig.write_html(os.path.join(output_dir, "plots", "confusion_matrix.html"), 
                      include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _generate_roc_curves_plot(self, y_true: pd.Series, y_proba: np.ndarray, 
                                 class_names: List[str], output_dir: str):
        """Generate ROC curves for each class."""
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_true)
        if y_true_binarized.shape[1] == 1:  # Binary case, expand to 2 columns
            y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
        
        fig = go.Figure()
        
        for i, class_name in enumerate(class_names):
            if i < y_proba.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
                auc_score = roc_auc_score(y_true_binarized[:, i], y_proba[:, i])
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{class_name} (AUC = {auc_score:.3f})',
                    line=dict(color=self.app_color_palette[i % len(self.app_color_palette)])
                ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves (One-vs-Rest)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        self._apply_color_theme(fig)
        fig.write_html(os.path.join(output_dir, "plots", "roc_curves.html"),
                      include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _generate_feature_importance_plot(self, model_wrapper, feature_names: List[str], output_dir: str):
        """Generate feature importance plot."""
        importance = model_wrapper.get_feature_importance()
        
        if importance is not None:
            # For multi-class LDA, take the average absolute coefficients across classes
            if len(importance.shape) > 1:
                importance = np.abs(importance).mean(axis=0)
            else:
                importance = np.abs(importance)
            
            fig = px.bar(
                x=feature_names,
                y=importance,
                title="Feature Importance (Absolute Coefficients)",
                color=importance,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_title="Features",
                yaxis_title="Importance Score",
                showlegend=False
            )
            
            self._apply_color_theme(fig)
            fig.write_html(os.path.join(output_dir, "plots", "feature_importance.html"),
                          include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _generate_prediction_confidence_plot(self, y_true: pd.Series, y_pred: np.ndarray,
                                           y_proba: np.ndarray, class_names: List[str], output_dir: str):
        """Generate prediction confidence distribution plot."""
        max_probabilities = np.max(y_proba, axis=1)
        correct_predictions = (y_true.values == y_pred)
        
        fig = go.Figure()
        
        # Correct predictions
        correct_probs = max_probabilities[correct_predictions]
        if len(correct_probs) > 0:
            fig.add_trace(go.Histogram(
                x=correct_probs,
                name="Correct Predictions",
                opacity=0.7,
                marker_color=self.app_color_palette[0],
                nbinsx=20
            ))
        
        # Incorrect predictions
        incorrect_probs = max_probabilities[~correct_predictions]
        if len(incorrect_probs) > 0:
            fig.add_trace(go.Histogram(
                x=incorrect_probs,
                name="Incorrect Predictions", 
                opacity=0.7,
                marker_color=self.app_color_palette[1],
                nbinsx=20
            ))
        
        fig.update_layout(
            title="Prediction Confidence Distribution",
            xaxis_title="Maximum Probability",
            yaxis_title="Count",
            barmode='overlay'
        )
        
        self._apply_color_theme(fig)
        fig.write_html(os.path.join(output_dir, "plots", "prediction_confidence.html"),
                      include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _generate_class_distribution_plot(self, y_train: pd.Series, y_test: pd.Series, 
                                        class_names: List[str], output_dir: str):
        """Generate class distribution comparison plot."""
        train_counts = y_train.value_counts().sort_index()
        test_counts = y_test.value_counts().sort_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=class_names,
            y=train_counts.values,
            name='Training Set',
            marker_color=self.app_color_palette[0]
        ))
        
        fig.add_trace(go.Bar(
            x=class_names,
            y=test_counts.values,
            name='Test Set',
            marker_color=self.app_color_palette[1]
        ))
        
        fig.update_layout(
            title="Class Distribution: Train vs Test Sets",
            xaxis_title="Species",
            yaxis_title="Count",
            barmode='group'
        )
        
        self._apply_color_theme(fig)
        fig.write_html(os.path.join(output_dir, "plots", "class_distribution.html"),
                      include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def _apply_color_theme(self, fig):
        """Apply consistent color theme to plots."""
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            font=dict(color='#8B5CF6', size=12),  # App's purple color for text
            title_font=dict(color='#7C3AED', size=16),  # Slightly darker purple for titles
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',  # Purple-tinted grid
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),  # Purple tick labels
                title_font=dict(color='#7C3AED', size=12)  # Darker purple axis titles
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',  # Purple-tinted grid
                zerolinecolor='rgba(139,92,246,0.3)', 
                tickfont=dict(color='#8B5CF6', size=11),  # Purple tick labels
                title_font=dict(color='#7C3AED', size=12)  # Darker purple axis titles
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))  # Purple legend
        )

    def save_metrics_json(self, metrics: Dict[str, Any], output_dir: str):
        """Save evaluation metrics to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        metrics_clean = convert_numpy_types(metrics)
        
        with open(os.path.join(output_dir, "general_artifacts", "evaluation_metrics.json"), 'w') as f:
            json.dump(metrics_clean, f, indent=2)
