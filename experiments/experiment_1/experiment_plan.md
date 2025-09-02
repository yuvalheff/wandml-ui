# Experiment 1: Baseline LDA Classification

## Overview
This experiment implements a Linear Discriminant Analysis (LDA) baseline for the Iris species classification task. Based on exploration experiments, LDA achieved perfect macro-AUC performance and is theoretically well-suited for this linearly separable multi-class problem.

## Task Details
- **Task Type**: Multi-class classification
- **Target Variable**: Species (Iris-setosa, Iris-versicolor, Iris-virginica)
- **Evaluation Metric**: Macro-averaged AUC
- **Dataset**: 120 training samples, 30 test samples, perfectly balanced classes

## Preprocessing Steps

### Data Loading
- Load training set from: `/Users/yuvalheffetz/ds-agent-projects/session_d4b0eb9f-8fbe-4190-b004-6f780e178fe3/data/train_set.csv`
- Load test set from: `/Users/yuvalheffetz/ds-agent-projects/session_d4b0eb9f-8fbe-4190-b004-6f780e178fe3/data/test_set.csv`
- Use feature columns: `["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]`
- Drop ID column: `["Id"]`

### Feature Scaling
- **Method**: StandardScaler
- **Rationale**: Exploration experiments showed StandardScaler provides optimal performance. Features have different natural scales (sepal width: 2.0-4.4 cm vs petal length: 1.1-6.9 cm)
- **Application**: Fit scaler on training set, transform both training and test sets
- **Features to scale**: All 4 numerical features

### Target Encoding  
- **Method**: LabelEncoder for sklearn compatibility
- **Classes**: ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

## Feature Engineering Steps

### Strategy: Use Original Features
- **Approach**: Use the original 4 botanical measurements without additional feature engineering
- **Rationale**: Exploration experiments demonstrated that original features achieve near-perfect performance (98.3% accuracy, 1.00 macro-AUC). Feature engineering provided minimal benefit and may introduce unnecessary complexity.

### Rejected Approaches
1. **Ratio Features**: Sepal/Petal length-to-width ratios showed no improvement
2. **Area Features**: Sepal/Petal area approximations showed no improvement  
3. **Polynomial Features**: Would add unnecessary complexity given perfect baseline performance

## Model Selection Steps

### Primary Algorithm: Linear Discriminant Analysis (LDA)
- **Parameters**:
  - `solver='svd'` (default, handles small dataset well)
  - `store_covariance=True` (for analysis purposes)
- **Rationale**: 
  - Achieved perfect macro-AUC (1.00) in exploration experiments
  - Theoretically well-suited for linearly separable multi-class problems
  - Provides probabilistic outputs needed for AUC calculation
  - Computationally efficient and interpretable

### Backup Algorithms
1. **LogisticRegression**: `max_iter=1000, multi_class='ovr'` (achieved 99.8% macro-AUC)
2. **SVM Linear**: `kernel='linear', probability=True` (achieved 97.5% accuracy)

### Cross-Validation
- **Method**: 5-fold Stratified Cross-Validation
- **Configuration**: `shuffle=True, random_state=42`
- **Purpose**: Model selection and performance estimation

## Evaluation Strategy

### Primary Metric: Macro-Averaged AUC
- **Calculation**: One-vs-rest approach, average AUC across all classes
- **Target**: >= 0.95 on test set

### Secondary Metrics
- Accuracy, Precision (macro), Recall (macro), F1-score (macro)

### Comprehensive Analysis Components

1. **Confusion Matrix Analysis**
   - Purpose: Identify which species pairs are most difficult to distinguish
   - Expected insight: Verify if Iris-setosa is perfectly separable

2. **Class-wise AUC Analysis**  
   - Calculate AUC for each one-vs-rest classification
   - Purpose: Understand performance per species

3. **Feature Importance Analysis**
   - Method: LDA feature weights/coefficients
   - Purpose: Identify which measurements contribute most to classification
   - Expected insight: Petal measurements likely more important based on EDA

4. **Decision Boundary Visualization**
   - Method: 2D projection using LDA components
   - Purpose: Visualize species separation in transformed space

5. **Prediction Confidence Analysis**
   - Method: Analyze probability distributions for correct vs incorrect predictions
   - Purpose: Understand model confidence and potential failure modes

6. **Per-Species Performance Analysis**
   - Metrics: Precision, Recall, F1-score per class
   - Purpose: Ensure balanced performance across all species

### Validation Approach
- **Model Selection**: 5-fold stratified cross-validation
- **Final Evaluation**: Holdout test set (30 samples, 10 per class)
- **Confidence Intervals**: 1000 bootstrap samples for test performance confidence

## Expected Outputs

### Model Artifacts
- `trained_lda_model.pkl` - Serialized trained LDA model
- `feature_scaler.pkl` - Fitted StandardScaler
- `label_encoder.pkl` - Fitted LabelEncoder

### Evaluation Results
- `macro_auc_score.json` - Primary metric result
- `classification_report.json` - Comprehensive classification metrics
- `confusion_matrix.png` - Visual confusion matrix
- `feature_importance_plot.png` - LDA feature weights visualization
- `decision_boundary_plot.png` - 2D decision boundary visualization  
- `class_auc_scores.json` - Individual class AUC scores

### Analysis Reports
- `model_performance_report.md` - Detailed performance analysis
- `feature_analysis_report.md` - Feature importance and contribution analysis

## Success Criteria

### Primary Success Criteria
- **Macro-averaged AUC >= 0.95** on test set

### Secondary Success Criteria
- Accuracy >= 0.90 on test set
- All individual classes achieve AUC >= 0.90  
- No class has F1-score < 0.85
- Model training completes without errors

## Implementation Notes

### Dataset Characteristics
- Perfectly balanced 3-class problem (40 samples per class in training)
- 4 numerical features, no missing values
- Clean, high-quality botanical measurements

### Computational Requirements
- **Complexity**: Low - linear model with small dataset
- **Expected Runtime**: < 1 minute for full pipeline
- **Memory**: Minimal requirements

### Reproducibility
- Set `random_state=42` for all stochastic components
- Use fixed train/test splits (already provided)
- Document all package versions used

## Domain-Specific Considerations

### Botanical Context  
- Features represent real botanical measurements from Fisher's 1936 study
- Iris-setosa is known to be linearly separable from other species
- Petal measurements typically more discriminative than sepal measurements

### Classification Challenge
- Primary challenge: Distinguishing Iris-versicolor from Iris-virginica  
- Secondary challenge: Ensuring robust performance across all measurement ranges
- Tertiary challenge: Maintaining interpretability for botanical insights