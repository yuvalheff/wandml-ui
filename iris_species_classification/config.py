from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


class ConfigParsingFailed(Exception):
    pass


@dataclass
class DataConfig:
    dataset_name: str
    feature_columns: List[str]
    target_column: str
    drop_columns: List[str]


@dataclass
class FeaturesConfig:
    apply_scaling: bool
    scaling_method: str


@dataclass
class ModelEvalConfig:
    cv_folds: int
    cv_shuffle: bool
    random_state: int
    primary_metric: str
    secondary_metrics: List[str]


@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any]


@dataclass
class Config:
    data_prep: DataConfig
    feature_prep: FeaturesConfig
    model_evaluation: ModelEvalConfig
    model: ModelConfig

    @staticmethod
    def from_yaml(config_file: str):
        with open(config_file, 'r', encoding='utf-8') as stream:
            try:
                config_data = yaml.safe_load(stream)
                return Config(
                    data_prep=DataConfig(**config_data['data_prep']),
                    feature_prep=FeaturesConfig(**config_data['feature_prep']),
                    model_evaluation=ModelEvalConfig(**config_data['model_evaluation']),
                    model=ModelConfig(**config_data['model'])
                )
            except (yaml.YAMLError, OSError) as e:
                raise ConfigParsingFailed from e