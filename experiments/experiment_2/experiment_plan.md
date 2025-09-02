# Experiment 2: Random Forest with Feature Engineering

## Overview
**Experiment Name**: Random Forest with Feature Engineering  
**Task Type**: Multi-class Classification  
**Target Variable**: Species  
**Primary Metric**: Macro-averaged AUC (target ≥ 0.95)  
**Iteration Focus**: Introduce meaningful feature engineering to capture morphological relationships while maintaining interpretability

## Background & Context

### Previous Iteration Results
- **Iteration 1 (LDA Baseline)**: Achieved perfect performance (1.0 macro-AUC, 100% accuracy)
- **Challenge**: Such perfect performance may indicate overfitting to this specific dataset split
- **Opportunity**: Test robustness through different modeling approaches and feature engineering

### Exploration Findings
Based on comprehensive exploration experiments:
- **SVM/MLP**: Highest individual performance (AUC ~0.9967, Accuracy ~0.9667)
- **Random Forest with Feature Engineering**: Strong performance (AUC 0.9933, Accuracy 0.9667) with interpretability
- **Key Insight**: Engineered features (PetalArea, PetalRatio) showed high importance in species discrimination

## Data Preprocessing

### Data Loading
- **Train Set**: `/data/train_set.csv` (120 samples, 40 per class)
- **Test Set**: `/data/test_set.csv` (30 samples, 10 per class)
- **Feature Columns**: `['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']`
- **Target Column**: `Species`

### Scaling Strategy
- **Method**: StandardScaler
- **Application**: Apply after feature engineering to all features (original + engineered)
- **Rationale**: Ensures equal contribution potential from all features, especially important for engineered features with different scales

## Feature Engineering

### Engineered Features (6 additional features)

1. **SepalRatio** = SepalLengthCm / SepalWidthCm
   - *Biological meaning*: Captures sepal shape proportions

2. **PetalRatio** = PetalLengthCm / PetalWidthCm
   - *Biological meaning*: Captures petal shape proportions

3. **SepalArea** = SepalLengthCm × SepalWidthCm
   - *Biological meaning*: Approximate sepal surface area

4. **PetalArea** = PetalLengthCm × PetalWidthCm
   - *Biological meaning*: Approximate petal surface area (**highest importance in exploration**)

5. **TotalLength** = SepalLengthCm + PetalLengthCm
   - *Biological meaning*: Overall flower length

6. **TotalWidth** = SepalWidthCm + PetalWidthCm
   - *Biological meaning*: Overall flower width

### Feature Engineering Rationale
- Captures **morphological relationships** beyond individual measurements
- Provides **biologically meaningful** interpretations
- **PetalArea** emerged as most discriminative feature in exploration
- Maintains **numerical continuity** suitable for Random Forest

## Model Selection

### Algorithm: RandomForestClassifier
**Rationale for Selection**:
- Strong performance with engineered features (AUC 0.9933 in exploration)
- Natural handling of feature interactions
- Built-in feature importance for interpretability
- Robust to overfitting with proper hyperparameters

### Hyperparameters
- **n_estimators**: 100 (balance performance/efficiency)
- **max_depth**: 10 (handle additional engineered features)
- **min_samples_split**: 2
- **min_samples_leaf**: 1
- **random_state**: 42 (reproducibility)

### Cross-Validation
- **Strategy**: 5-fold stratified cross-validation
- **Purpose**: Robust performance estimation across all three species
- **Benefit**: Maintains class balance in all folds

## Evaluation Strategy

### Primary Metrics
- **Macro-averaged AUC**: Primary target ≥ 0.95
- **Accuracy**: Overall classification correctness
- **Precision (Macro)**: Average positive prediction accuracy
- **Recall (Macro)**: Average sensitivity across classes
- **F1-Score (Macro)**: Harmonic mean of precision and recall

### Comprehensive Analysis Outputs

#### 1. Performance Visualizations
- **confusion_matrix.html**: Interactive confusion matrix with error analysis
- **roc_curves.html**: Individual class ROC curves
- **prediction_confidence.html**: Model confidence distribution analysis
- **class_performance.html**: Per-class detailed performance breakdown

#### 2. Model Interpretability
- **feature_importance.html**: Ranked importance of all features (original + engineered)
- **feature_correlation.html**: Correlation matrix showing feature relationships
- Focus on understanding biological significance of top features

#### 3. Robustness Assessment
- **Cross-validation results**: Performance stability across folds
- **Error analysis**: Patterns in misclassified samples
- **Comparison with LDA baseline**: Performance differences and trade-offs

### Diagnostic Questions to Address
1. Which engineered features contribute most to species discrimination?
2. Are there consistent error patterns across different species?
3. How does prediction confidence vary across species?
4. What biological insights emerge from feature importance rankings?

## Expected Outputs

### Model Artifacts
- Trained RandomForestClassifier model
- Fitted StandardScaler transformer
- Cross-validation performance metrics

### Analysis Reports
- Comprehensive classification report with all metrics
- Feature importance analysis with biological interpretation
- Error analysis of misclassified samples
- Performance comparison with LDA baseline

### Success Criteria
- **Performance**: Macro-averaged AUC ≥ 0.95
- **Interpretability**: Clear biological insights from feature importance
- **Robustness**: Stable cross-validation performance (low variance)
- **Understanding**: Identify key morphological discriminators

## Implementation Notes

### Data Handling
- No missing values or data quality issues identified
- Maintain stratified sampling throughout analysis
- All features treated as continuous numerical variables

### Quality Assurance
- Validate engineered features for reasonable ranges
- Ensure feature scaling applies correctly to all features
- Cross-check performance metrics calculation
- Verify reproducibility with fixed random seeds

## Biological Context & Expected Insights

### Species Characteristics
- **Iris-setosa**: Typically smaller petals, easily separable
- **Iris-versicolor**: Intermediate characteristics  
- **Iris-virginica**: Larger overall measurements

### Expected Feature Importance Hierarchy
1. **PetalArea**: Likely highest (confirmed in exploration)
2. **PetalWidthCm**: Strong individual discriminator
3. **PetalLengthCm**: Traditional strong feature
4. **PetalRatio**: Shape information
5. **SepalRatio**: Additional shape context
6. Lower importance: Total measurements and sepal individual measurements

This experiment design balances **performance optimization** with **biological interpretability**, providing actionable insights into iris species classification while testing robustness beyond the perfect LDA baseline.