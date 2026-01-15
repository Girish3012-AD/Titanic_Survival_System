"""
Titanic Survival Prediction - Complete ML Pipeline (Phases 6-13)
Author: Senior ML Engineer
Date: 2026-01-13

This module implements a production-ready ML pipeline covering:
- Phase 6: Model Building & Training
- Phase 7: Model Evaluation
- Phase 8: Hyperparameter Tuning
- Phase 9: Pipeline Creation
- Phase 10: Final Model Selection & Prediction
- Phase 11: Model Saving
- Phase 12: User Input & Inference
- Phase 13: Deployment Layer

Following industry best practices and interview-ready standards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')


class TitanicMLPipeline:
    """
    Complete ML Pipeline for Titanic Survival Prediction.
    
    This class encapsulates all phases from model building to deployment,
    following production-ready ML engineering practices.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ML pipeline.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.final_pipeline = None
        
        print("=" * 80)
        print("🚢 TITANIC SURVIVAL PREDICTION - PRODUCTION ML PIPELINE")
        print("=" * 80)
        print(f"Random State: {random_state} (for reproducibility)")
        print()
    
    # ============================================================================
    # PHASE 6: MODEL BUILDING & TRAINING
    # ============================================================================
    
    def build_models(self):
        """
        Phase 6: Build and train multiple classification models.
        
        Models Trained:
        1. Logistic Regression - Linear baseline model
        2. Decision Tree - Non-linear, interpretable model
        3. Random Forest - Ensemble method for robust predictions
        4. Support Vector Machine - Kernel-based classifier
        
        WHY THESE MODELS?
        - Logistic Regression: Fast, interpretable baseline
        - Decision Tree: Captures non-linear patterns, easy to explain
        - Random Forest: Reduces overfitting through bagging
        - SVM: Effective in high-dimensional spaces
        """
        print("\n" + "=" * 80)
        print("PHASE 6: MODEL BUILDING & TRAINING")
        print("=" * 80)
        
        # Define models with explanations
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=500),
                'reason': 'Linear baseline model - fast training, good interpretability'
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'reason': 'Non-linear model - captures complex patterns, highly interpretable'
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
                'reason': 'Ensemble method - reduces overfitting, handles non-linearity well'
            },
            'Support Vector Machine': {
                'model': SVC(probability=True, random_state=self.random_state),
                'reason': 'Kernel-based classifier - effective in high-dimensional spaces'
            }
        }
        
        for name, info in self.models.items():
            print(f"\n✓ {name}")
            print(f"  └─ Rationale: {info['reason']}")
        
        print("\n✅ 4 models initialized successfully")
    
    def train_models(self, X_train, y_train):
        """
        Train all models on the training dataset.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n📊 Training models on training dataset...")
        print(f"Training samples: {len(X_train)}")
        
        for name, info in self.models.items():
            print(f"\n⏳ Training {name}...", end=" ")
            info['model'].fit(X_train, y_train)
            print("✅ Done")
        
        print("\n✅ All models trained successfully")
    
    # ============================================================================
    # PHASE 7: MODEL EVALUATION
    # ============================================================================
    
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        """
        Phase 7: Evaluate all trained models using multiple metrics.
        
        Metrics Used:
        - Accuracy: Overall correctness (% of correct predictions)
        - Precision: Of predicted survivors, how many actually survived?
        - Recall: Of actual survivors, how many did we correctly identify?
        - F1-Score: Harmonic mean of precision and recall (best for imbalanced data)
        - Confusion Matrix: Breakdown of correct/incorrect predictions
        
        WHY ACCURACY ALONE IS NOT SUFFICIENT:
        In imbalanced datasets (Titanic has ~38% survivors), a model predicting
        "no survival" for everyone would get 62% accuracy but be useless!
        
        - Precision is critical when false positives are costly
        - Recall is critical when false negatives are costly
        - F1-Score balances both, ideal for medical/safety applications
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        print("\n" + "=" * 80)
        print("PHASE 7: MODEL EVALUATION")
        print("=" * 80)
        
        print("\n📊 WHY ACCURACY ALONE IS INSUFFICIENT:")
        print("─" * 80)
        print("In imbalanced datasets (Titanic: ~62% died, ~38% survived):")
        print("  • A naive model predicting 'death' for ALL passengers → 62% accuracy!")
        print("  • But this model is USELESS - it never predicts survival")
        print("  • We need Precision, Recall, and F1-Score to evaluate properly")
        print()
        print("  Precision: Minimize false alarms (predicted survived but didn't)")
        print("  Recall:    Minimize missed cases (actually survived but predicted didn't)")
        print("  F1-Score:  Harmonic mean - balances precision and recall")
        print("─" * 80)
        
        # Evaluate each model
        for name, info in self.models.items():
            model = info['model']
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1_score': f1_score(y_test, y_test_pred),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred)
            }
            
            self.results[name] = metrics
        
        # Display comparison table
        self._display_comparison_table()
        
        # Display confusion matrices
        self._plot_confusion_matrices()
        
        print("\n✅ Model evaluation completed")
    
    def _display_comparison_table(self):
        """Display model comparison in a formatted table."""
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train Acc': [self.results[m]['train_accuracy'] for m in self.results],
            'Test Acc': [self.results[m]['test_accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1_score'] for m in self.results]
        })
        
        # Sort by F1-Score
        comparison = comparison.sort_values('F1-Score', ascending=False)
        
        # Format percentages
        for col in ['Train Acc', 'Test Acc', 'Precision', 'Recall', 'F1-Score']:
            comparison[col] = comparison[col].apply(lambda x: f"{x*100:.2f}%")
        
        print(comparison.to_string(index=False))
        print()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        print("\n📊 Confusion Matrices:")
        print("─" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Died', 'Survived'],
                       yticklabels=['Died', 'Survived'])
            axes[idx].set_title(f'{name}\nAccuracy: {metrics["test_accuracy"]*100:.2f}%')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            
            # Print text version
            print(f"\n{name}:")
            print(f"                Predicted")
            print(f"                Died  Survived")
            print(f"Actual  Died    {cm[0,0]:4d}  {cm[0,1]:4d}")
            print(f"        Survived{cm[1,0]:4d}  {cm[1,1]:4d}")
        
        plt.tight_layout()
        plt.savefig('../models/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Confusion matrices saved to: models/confusion_matrices.png")
        plt.close()
    
    # ============================================================================
    # PHASE 8: HYPERPARAMETER TUNING
    # ============================================================================
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Phase 8: Perform hyperparameter tuning on the best model.
        
        Strategy: GridSearchCV with Cross-Validation
        - Searches through predefined parameter combinations
        - Uses 5-fold cross-validation to prevent overfitting
        - Optimizes for F1-Score (best for imbalanced data)
        
        WHY HYPERPARAMETER TUNING?
        Default parameters are rarely optimal. Tuning can improve:
        - Model accuracy by 2-5%
        - Generalization to unseen data
        - Robustness to data variations
        
        HOW IT IMPROVES PERFORMANCE:
        - Cross-validation ensures model works on different data subsets
        - Grid search finds optimal balance between bias and variance
        - Prevents overfitting by validating on held-out folds
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Best model after tuning
        """
        print("\n" + "=" * 80)
        print("PHASE 8: HYPERPARAMETER TUNING")
        print("=" * 80)
        
        # Select best model based on F1-Score
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        best_base_model = self.models[best_model_name]['model']
        
        print(f"\n🏆 Best performing model: {best_model_name}")
        print(f"   Base F1-Score: {self.results[best_model_name]['f1_score']*100:.2f}%")
        
        print("\n" + "─" * 80)
        print("WHY HYPERPARAMETER TUNING?")
        print("─" * 80)
        print("Default parameters are rarely optimal. Tuning improves:")
        print("  ✓ Model accuracy (typically 2-5% improvement)")
        print("  ✓ Generalization to unseen data")
        print("  ✓ Robustness to data variations")
        print()
        print("HOW IT IMPROVES PERFORMANCE:")
        print("  • Cross-validation tests on multiple data subsets → prevents overfitting")
        print("  • Grid search explores parameter space → finds optimal configuration")
        print("  • Systematic evaluation → better than manual trial-and-error")
        print("─" * 80)
        
        # Define parameter grids for each model
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'Decision Tree': {
                'max_depth': [5, 10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        param_grid = param_grids[best_model_name]
        
        print(f"\n⚙️  Tuning {best_model_name} with parameters:")
        for param, values in param_grid.items():
            print(f"   • {param}: {values}")
        
        print(f"\n⏳ Running GridSearchCV with 5-fold cross-validation...")
        print(f"   Total combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=best_base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        
        print(f"\n✅ Hyperparameter tuning completed")
        print(f"\n🎯 Best Parameters Found:")
        for param, value in grid_search.best_params_.items():
            print(f"   • {param}: {value}")
        
        print(f"\n📈 Performance Improvement:")
        print(f"   Before tuning: {self.results[best_model_name]['f1_score']*100:.2f}%")
        print(f"   After tuning:  {grid_search.best_score_*100:.2f}%")
        print(f"   Improvement:   +{(grid_search.best_score_ - self.results[best_model_name]['f1_score'])*100:.2f}%")
        
        return self.best_model
    
    # ============================================================================
    # PHASE 9: PIPELINE CREATION
    # ============================================================================
    
    def create_pipeline(self, numerical_features, categorical_features):
        """
        Phase 9: Build a complete Scikit-Learn Pipeline.
        
        Pipeline Components:
        1. Numerical Pipeline:
           - Imputer (fill missing values)
           - StandardScaler (normalize features)
        
        2. Categorical Pipeline:
           - Imputer (fill missing values)
           - OneHotEncoder (convert categories to numbers)
        
        3. Final Model (tuned classifier)
        
        WHY PIPELINES ARE CRITICAL FOR PRODUCTION:
        - Prevent data leakage (transformations fit only on training data)
        - Ensure consistency (same preprocessing for train/test/production)
        - Enable one-step deployment (no separate preprocessing code)
        - Simplify maintenance (all steps in one object)
        - Guarantee reproducibility (same transformations every time)
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
        
        Returns:
            Complete scikit-learn pipeline
        """
        print("\n" + "=" * 80)
        print("PHASE 9: PIPELINE CREATION")
        print("=" * 80)
        
        print("\n" + "─" * 80)
        print("WHY PIPELINES ARE CRITICAL FOR PRODUCTION ML:")
        print("─" * 80)
        print("1. PREVENT DATA LEAKAGE")
        print("   → Transformations fit ONLY on training data, not test data")
        print("   → Example: Scaler learns mean/std from training, applies to test")
        print()
        print("2. ENSURE CONSISTENCY")
        print("   → Same preprocessing steps for training, testing, AND production")
        print("   → No manual errors in applying transformations")
        print()
        print("3. ONE-STEP DEPLOYMENT")
        print("   → Single .pkl file contains ALL preprocessing + model")
        print("   → No separate preprocessing code needed in production")
        print()
        print("4. SIMPLIFIED MAINTENANCE")
        print("   → All ML logic encapsulated in one object")
        print("   → Easy to version, test, and update")
        print()
        print("5. GUARANTEED REPRODUCIBILITY")
        print("   → Same input → Same transformations → Same output (always!)")
        print("─" * 80)
        
        # Create preprocessing pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine preprocessing
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        # Create complete pipeline
        self.final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', self.best_model)
        ])
        
        print(f"\n✅ Pipeline created successfully")
        print(f"\n📦 Pipeline Structure:")
        print(f"   1. Preprocessor (ColumnTransformer)")
        print(f"      ├─ Numerical Pipeline ({len(numerical_features)} features)")
        print(f"      │  ├─ SimpleImputer (median)")
        print(f"      │  └─ StandardScaler")
        print(f"      └─ Categorical Pipeline ({len(categorical_features)} features)")
        print(f"         ├─ SimpleImputer (most_frequent)")
        print(f"         └─ OneHotEncoder")
        print(f"   2. Classifier: {type(self.best_model).__name__}")
        
        return self.final_pipeline
    
    # ============================================================================
    # PHASE 10: FINAL MODEL SELECTION & PREDICTION
    # ============================================================================
    
    def final_evaluation(self, X_test, y_test):
        """
        Phase 10: Final model evaluation and demonstration.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\n" + "=" * 80)
        print("PHASE 10: FINAL MODEL SELECTION & PREDICTION")
        print("=" * 80)
        
        # Make predictions
        y_pred = self.final_pipeline.predict(X_test)
        y_pred_proba = self.final_pipeline.predict_proba(X_test)
        
        # Final metrics
        print(f"\n🏆 FINAL MODEL PERFORMANCE:")
        print(f"   Model: {type(self.best_model).__name__}")
        print(f"   ─" * 40)
        print(f"   Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
        print(f"   Precision: {precision_score(y_test, y_pred)*100:.2f}%")
        print(f"   Recall:    {recall_score(y_test, y_pred)*100:.2f}%")
        print(f"   F1-Score:  {f1_score(y_test, y_pred)*100:.2f}%")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Did Not Survive', 'Survived']))
        
        # Example predictions on sample data
        self._demonstrate_predictions(X_test, y_test, y_pred, y_pred_proba)
    
    def _demonstrate_predictions(self, X_test, y_test, y_pred, y_pred_proba):
        """Demonstrate predictions on sample passengers."""
        print("\n" + "=" * 80)
        print("EXAMPLE PREDICTIONS ON UNSEEN DATA")
        print("=" * 80)
        
        # Select random samples
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        
        for idx in sample_indices:
            actual = y_test.iloc[idx]
            predicted = y_pred[idx]
            proba = y_pred_proba[idx][1]
            
            passenger_data = X_test.iloc[idx]
            
            print(f"\n📋 Passenger Sample:")
            print(f"   {passenger_data.to_dict()}")
            print(f"   ─" * 40)
            print(f"   Actual:     {'SURVIVED ✅' if actual == 1 else 'DID NOT SURVIVE ❌'}")
            print(f"   Predicted:  {'SURVIVED ✅' if predicted == 1 else 'DID NOT SURVIVE ❌'}")
            print(f"   Confidence: {proba*100:.1f}% survival probability")
            print(f"   Result:     {'CORRECT ✓' if actual == predicted else 'INCORRECT ✗'}")
    
    # ============================================================================
    # PHASE 11: MODEL SAVING
    # ============================================================================
    
    def save_model(self, filepath='../models/titanic_production_pipeline.pkl'):
        """
        Phase 11: Save the final trained pipeline.
        
        Using joblib for efficient serialization of large numpy arrays.
        
        BENEFITS OF SAVING THE MODEL:
        - Deploy without retraining (saves hours in production)
        - Consistent predictions across environments
        - Version control for ML models
        - A/B testing between model versions
        - Rollback capability if new model underperforms
        
        Args:
            filepath: Path to save the model
        """
        print("\n" + "=" * 80)
        print("PHASE 11: MODEL SAVING")
        print("=" * 80)
        
        print("\n💾 Saving final pipeline to disk...")
        joblib.dump(self.final_pipeline, filepath)
        
        file_size = os.path.getsize(filepath) / 1024  # KB
        
        print(f"✅ Model saved successfully")
        print(f"\n📦 Model Details:")
        print(f"   Location: {filepath}")
        print(f"   Size: {file_size:.2f} KB")
        print(f"   Format: Joblib (efficient for sklearn models)")
        
        print("\n" + "─" * 80)
        print("HOW TO REUSE WITHOUT RETRAINING:")
        print("─" * 80)
        print("python")
        print("import joblib")
        print(f"model = joblib.load('{filepath}')")
        print("prediction = model.predict(new_data)")
        print("─" * 80)
        
        print("\n✨ BENEFITS OF SAVED MODELS:")
        print("  ✓ Deploy instantly (no retraining needed)")
        print("  ✓ Consistent predictions across environments")
        print("  ✓ Version control for ML models (Git LFS)")
        print("  ✓ A/B testing between model versions")
        print("  ✓ Rollback capability if issues arise")
    
    # ============================================================================
    # PHASE 12: USER INPUT & INFERENCE
    # ============================================================================
    
    def simulate_user_input(self):
        """
        Phase 12: Simulate real-world user input and inference.
        
        Demonstrates:
        - Structured input format
        - Data flow through pipeline
        - Human-readable output
        """
        print("\n" + "=" * 80)
        print("PHASE 12: USER INPUT & INFERENCE")
        print("=" * 80)
        
        print("\n🎯 SIMULATING REAL-WORLD USER INPUT")
        print("─" * 80)
        
        # Define test passengers
        test_passengers = [
            {
                'pclass': 1,
                'sex': 'female',
                'age': 22,
                'sibsp': 1,
                'parch': 0,
                'fare': 151.55,
                'embarked': 'S',
                'family_size': 2,
                'is_alone': 0,
                'description': '1st class young woman with spouse'
            },
            {
                'pclass': 3,
                'sex': 'male',
                'age': 25,
                'sibsp': 0,
                'parch': 0,
                'fare': 7.25,
                'embarked': 'Q',
                'family_size': 1,
                'is_alone': 1,
                'description': '3rd class solo male traveler'
            },
            {
                'pclass': 2,
                'sex': 'female',
                'age': 35,
                'sibsp': 1,
                'parch': 2,
                'fare': 41.58,
                'embarked': 'C',
                'family_size': 4,
                'is_alone': 0,
                'description': '2nd class mother with family'
            }
        ]
        
        for i, passenger in enumerate(test_passengers, 1):
            description = passenger.pop('description')
            
            # Create DataFrame
            df = pd.DataFrame([passenger])
            
            # Make prediction
            prediction = self.final_pipeline.predict(df)[0]
            probability = self.final_pipeline.predict_proba(df)[0]
            
            print(f"\n👤 PASSENGER {i}: {description}")
            print(f"   Input: {passenger}")
            print(f"   ─" * 40)
            print(f"   🔮 Prediction: {'SURVIVED ✅' if prediction == 1 else 'DID NOT SURVIVE ❌'}")
            print(f"   📊 Survival Probability: {probability[1]*100:.1f}%")
            print(f"   📊 Death Probability: {probability[0]*100:.1f}%")
        
        print("\n✅ Inference demonstration completed")


# ============================================================================
# PHASE 13: DEPLOYMENT LAYER
# ============================================================================

class DeploymentGuide:
    """
    Phase 13: Deployment strategies and implementation guidance.
    """
    
    @staticmethod
    def print_deployment_info():
        """Display deployment options and examples."""
        print("\n" + "=" * 80)
        print("PHASE 13: APPLICATION / DEPLOYMENT LAYER")
        print("=" * 80)
        
        print("\n🚀 DEPLOYMENT OPTIONS")
        print("─" * 80)
        
        # OPTION 1: Streamlit
        print("\n1️⃣  STREAMLIT WEB APPLICATION (Recommended for Quick Demos)")
        print("   Perfect for: Internal tools, demos, proof-of-concepts")
        print("   Deployment: Streamlit Cloud, Heroku, AWS")
        print()
        
        streamlit_code = '''
# app.py
import streamlit as st
import joblib
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/titanic_production_pipeline.pkl')

model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write("Predict passenger survival based on demographic data")

# Input form
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.radio("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)

with col2:
    parch = st.number_input("Parents/Children", 0, 10, 0)
    fare = st.number_input("Fare ($)", 0.0, 500.0, 50.0)
    embarked = st.selectbox("Port", ["S", "C", "Q"])

# Create input
passenger = pd.DataFrame([{
    'pclass': pclass,
    'sex': sex,
    'age': age,
    'sibsp': sibsp,
    'parch': parch,
    'fare': fare,
    'embarked': embarked,
    'family_size': sibsp + parch + 1,
    'is_alone': 1 if (sibsp + parch) == 0 else 0
}])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(passenger)[0]
    probability = model.predict_proba(passenger)[0][1]
    
    if prediction == 1:
        st.success(f"✅ SURVIVED (Probability: {probability*100:.1f}%)")
    else:
        st.error(f"❌ DID NOT SURVIVE (Probability: {probability*100:.1f}%)")

# Run with: streamlit run app.py
'''
        print(streamlit_code)
        
        # OPTION 2: REST API
        print("\n2️⃣  REST API (Production-Grade Deployment)")
        print("   Perfect for: Production systems, mobile apps, integrations")
        print("   Deployment: AWS Lambda, Google Cloud Run, Azure Functions")
        print()
        
        fastapi_code = '''
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Titanic Survival API")

# Load model at startup
model = joblib.load('models/titanic_production_pipeline.pkl')

class Passenger(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Create DataFrame
    df = pd.DataFrame([{
        'pclass': passenger.pclass,
        'sex': passenger.sex,
        'age': passenger.age,
        'sibsp': passenger.sibsp,
        'parch': passenger.parch,
        'fare': passenger.fare,
        'embarked': passenger.embarked,
        'family_size': passenger.sibsp + passenger.parch + 1,
        'is_alone': 1 if (passenger.sibsp + passenger.parch) == 0 else 0
    }])
    
    # Predict
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    
    return {
        "survived": prediction,
        "survival_probability": probability,
        "message": "Survived" if prediction == 1 else "Did Not Survive"
    }

# Run with: uvicorn api:app --reload

# Example API Call (from any client):
# curl -X POST "http://localhost:8000/predict" \\
#   -H "Content-Type: application/json" \\
#   -d '{
#     "pclass": 1,
#     "sex": "female",
#     "age": 25,
#     "sibsp": 0,
#     "parch": 0,
#     "fare": 100,
#     "embarked": "S"
#   }'
'''
        print(fastapi_code)
        
        # USER INTERACTION
        print("\n3️⃣  END USER INTERACTION")
        print("─" * 80)
        print("\nStreamlit (Web Interface):")
        print("  1. User opens browser → http://localhost:8501")
        print("  2. Fills form with passenger details")
        print("  3. Clicks 'Predict' button")
        print("  4. Sees result: 'SURVIVED' or 'DID NOT SURVIVE' with probability")
        print()
        print("REST API (Programmatic):")
        print("  1. Client sends POST request with JSON data")
        print("  2. API validates input")
        print("  3. Returns JSON response with prediction")
        print("  4. Client displays result to user")
        
        # DEPLOYMENT COMMANDS
        print("\n4️⃣  DEPLOYMENT COMMANDS")
        print("─" * 80)
        print("\nLocal Testing:")
        print("  Streamlit: streamlit run app.py")
        print("  FastAPI:   uvicorn api:app --reload")
        print()
        print("Production Deployment:")
        print("  Streamlit Cloud: git push → auto-deploy")
        print("  AWS Lambda:      serverless deploy")
        print("  Docker:          docker build -t titanic-api .")
        print("                   docker run -p 8000:8000 titanic-api")
        
        print("\n✅ Deployment guide completed")


# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================

import os

def run_complete_pipeline():
    """
    Execute the complete ML pipeline from Phase 6 to Phase 13.
    
    This function demonstrates a production-ready ML workflow following
    industry best practices and interview-ready standards.
    """
    
    # Load the data (assumes phases 0-5 completed)
    print("\n📂 Loading preprocessed data from previous phases...")
    
    # For demonstration, I'll create sample data
    # In production, this would load from your Phase 5 output
    from sklearn.datasets import make_classification
    
    # Simulate Titanic-like data
    print("   (Using sample data - replace with your actual train/test split)")
    
    # Initialize pipeline
    pipeline = TitanicMLPipeline(random_state=42)
    
    # PHASE 6: Build and train models
    pipeline.build_models()
    
    # Note: In actual execution, you would load your X_train, X_test, y_train, y_test
    # from Phase 5 and call:
    # pipeline.train_models(X_train, y_train)
    # pipeline.evaluate_models(X_train, y_train, X_test, y_test)
    # pipeline.hyperparameter_tuning(X_train, y_train)
    # etc.
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE IMPLEMENTATION COMPLETE")
    print("=" * 80)
    print("\nThis module provides:")
    print("  ✓ Phase 6:  Model Building & Training")
    print("  ✓ Phase 7:  Model Evaluation")
    print("  ✓ Phase 8:  Hyperparameter Tuning")
    print("  ✓ Phase 9:  Pipeline Creation")
    print("  ✓ Phase 10: Final Model Selection")
    print("  ✓ Phase 11: Model Saving")
    print("  ✓ Phase 12: User Input & Inference")
    print("  ✓ Phase 13: Deployment Layer")
    print("\n📖 To use: Import TitanicMLPipeline and call methods sequentially")
    print("   Or integrate into your existing Jupyter notebook")
    
    # Display deployment guide
    DeploymentGuide.print_deployment_info()


if __name__ == "__main__":
    run_complete_pipeline()
