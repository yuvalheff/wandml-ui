import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import sklearn

from iris_species_classification.pipeline.feature_preprocessing import FeatureProcessor
from iris_species_classification.pipeline.data_preprocessing import DataProcessor
from iris_species_classification.pipeline.model import ModelWrapper
from iris_species_classification.config import Config
from iris_species_classification.model_pipeline import ModelPipeline
from experiment_scripts.evaluation import ModelEvaluator

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path, test_dataset_path, output_dir, seed=42):
        """
        Run the complete ML experiment pipeline.
        
        Parameters:
        train_dataset_path: Path to training dataset
        test_dataset_path: Path to test dataset  
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility
        
        Returns:
        Dict containing experiment results and MLflow model info
        """
        # Set random seed
        np.random.seed(seed)
        
        # Ensure output directories exist
        output_subdir = os.path.join(output_dir, "output")
        os.makedirs(os.path.join(output_subdir, "model_artifacts"), exist_ok=True)
        os.makedirs(os.path.join(output_subdir, "general_artifacts"), exist_ok=True)
        os.makedirs(os.path.join(output_subdir, "plots"), exist_ok=True)
        
        try:
            # Load data
            print("Loading datasets...")
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
            
            # Separate features and target
            X_train_raw = train_data.drop(columns=[self._config.data_prep.target_column])
            y_train_raw = train_data[self._config.data_prep.target_column]
            X_test_raw = test_data.drop(columns=[self._config.data_prep.target_column])
            y_test_raw = test_data[self._config.data_prep.target_column]
            
            # Get class names for evaluation
            class_names = sorted(y_train_raw.unique())
            
            # Initialize processors
            print("Initializing processors...")
            data_processor = DataProcessor(self._config.data_prep)
            feature_processor = FeatureProcessor(self._config.feature_prep)
            model_wrapper = ModelWrapper(self._config.model)
            
            # Fit and transform data
            print("Processing data...")
            data_processor.fit(X_train_raw, y_train_raw)
            X_train_processed = data_processor.transform(X_train_raw)
            X_test_processed = data_processor.transform(X_test_raw)
            
            # Transform target variables
            y_train = data_processor.transform_target(y_train_raw)
            y_test = data_processor.transform_target(y_test_raw)
            
            # Fit and transform features
            feature_processor.fit(X_train_processed)
            X_train = feature_processor.transform(X_train_processed)
            X_test = feature_processor.transform(X_test_processed)
            
            # Train model
            print("Training model...")
            model_wrapper.fit(X_train, y_train)
            
            # Evaluate model
            print("Evaluating model...")
            evaluator = ModelEvaluator(self._config.model_evaluation)
            metrics = evaluator.evaluate_model(
                model_wrapper, X_train, y_train, X_test, y_test, 
                class_names, output_subdir
            )
            
            # Save evaluation metrics
            evaluator.save_metrics_json(metrics, output_subdir)
            
            # Save individual model artifacts
            print("Saving model artifacts...")
            artifact_paths = []
            
            data_processor_path = os.path.join(output_subdir, "model_artifacts", "data_processor.pkl")
            data_processor.save(data_processor_path)
            artifact_paths.append("data_processor.pkl")
            
            feature_processor_path = os.path.join(output_subdir, "model_artifacts", "feature_processor.pkl")
            feature_processor.save(feature_processor_path)
            artifact_paths.append("feature_processor.pkl")
            
            model_wrapper_path = os.path.join(output_subdir, "model_artifacts", "trained_model.pkl")
            model_wrapper.save(model_wrapper_path)
            artifact_paths.append("trained_model.pkl")
            
            # Create and test the pipeline
            print("Creating MLflow pipeline...")
            pipeline = ModelPipeline(data_processor, feature_processor, model_wrapper, class_names)
            
            # Test pipeline with sample data to ensure it works
            sample_input = test_data.head(3).drop(columns=[self._config.data_prep.target_column])
            sample_predictions = pipeline.predict(sample_input)
            sample_probabilities = pipeline.predict_proba(sample_input)
            
            print(f"Pipeline test successful. Sample predictions: {sample_predictions}")
            
            # Define paths for MLflow model
            mlflow_model_path = os.path.join(output_subdir, "model_artifacts", "mlflow_model")
            relative_path_for_return = "output/model_artifacts/mlflow_model/"
            
            # Create model signature
            signature = infer_signature(sample_input, sample_predictions)
            
            # 1. Always save the model to the local path for harness validation
            print(f"💾 Saving model to local disk for harness: {mlflow_model_path}")
            mlflow.sklearn.save_model(
                pipeline,
                path=mlflow_model_path,
                signature=signature
            )
            
            # 2. If an MLflow run ID is provided, reconnect and log the model as an artifact
            active_run_id = "9163e71264b742f39d0e2fde12b2ddf4"
            logged_model_uri = None
            
            if active_run_id and active_run_id != 'None' and active_run_id.strip():
                print(f"✅ Active MLflow run ID '{active_run_id}' detected. Reconnecting to log model as an artifact.")
                try:
                    with mlflow.start_run(run_id=active_run_id):
                        logged_model_info = mlflow.sklearn.log_model(
                            pipeline,
                            artifact_path="model",
                            code_paths=["iris_species_classification"],
                            signature=signature
                        )
                        logged_model_uri = logged_model_info.model_uri
                        
                        # Also log key metrics
                        mlflow.log_metric("accuracy", metrics["accuracy"])
                        mlflow.log_metric("macro_averaged_auc", metrics["macro_averaged_auc"])
                        mlflow.log_metric("f1_macro", metrics["f1_macro"])
                        
                except Exception as e:
                    print(f"⚠️ Failed to log to MLflow run: {e}")
                    logged_model_uri = None
            else:
                print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
            
            # Get primary metric value
            primary_metric_value = metrics.get(self._config.model_evaluation.primary_metric, 0.0)
            
            print(f"Experiment completed successfully! Primary metric ({self._config.model_evaluation.primary_metric}): {primary_metric_value:.4f}")
            
            # Return results in expected format
            return {
                "metric_name": self._config.model_evaluation.primary_metric,
                "metric_value": float(primary_metric_value),
                "model_artifacts": artifact_paths,
                "mlflow_model_info": {
                    "model_path": relative_path_for_return,
                    "logged_model_uri": logged_model_uri,
                    "model_type": "sklearn",
                    "task_type": "classification",
                    "signature": signature.to_dict() if signature else None,
                    "input_example": sample_input.to_dict(orient='records'),
                    "python_model_class": "ModelPipeline",
                    "framework_version": sklearn.__version__
                }
            }
            
        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            raise e