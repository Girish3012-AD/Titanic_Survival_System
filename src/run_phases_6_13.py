"""
Complete Titanic ML Pipeline - Phases 6-13 Executable
Author: Senior ML Engineer
Date: 2026-01-13

This script executes ALL phases (6-13) in sequence with comprehensive output.
Run this file to see the complete ML engineering workflow in action.

Usage:
    python src/run_phases_6_13.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def print_header(text, char="=", width=80):
    """Print a formatted header."""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def print_section(text, char="─", width=80):
    """Print a formatted section."""
    print("\n" + char * width)
    print(text)
    print(char * width)


def main():
    """Execute the complete Phases 6-13 pipeline."""
    
    print_header("🚢 TITANIC SURVIVAL PREDICTION - PHASES 6-13", "═")
    print("\nProduction-Ready ML Pipeline Implementation")
    print("Author: Senior ML Engineer")
    print("Date: 2026-01-13")
    
    # =========================================================================
    # DATA PREPARATION (Recap of Phases 1-5)
    # =========================================================================
    
    print_header("PHASE 0-5 RECAP: DATA PREPARATION")
    
    print("\n⏳ Loading Titanic dataset...")
    df = sns.load_dataset('titanic')
    print(f"✅ Dataset loaded: {df.shape[0]} passengers, {df.shape[1]} features")
    
    # Select features
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target = 'survived'
    
    data = df[features + [target]].copy()
    
    print(f"\nSelected features: {len(features)}")
    print(f"Missing values:\n{data.isnull().sum()}")
    
    # Feature Engineering
    print("\n⚙️  Feature Engineering...")
    data['family_size'] = data['sibsp'] + data['parch'] + 1
    data['is_alone'] = (data['family_size'] == 1).astype(int)
    
    print("✅ Created features:")
    print("   • family_size: Total family members aboard")
    print("   • is_alone: Binary indicator for solo travelers")
    
    # Define feature types
    numerical_features = ['age', 'fare', 'sibsp', 'parch', 'family_size', 'is_alone']
    categorical_features = ['pclass', 'sex', 'embarked']
    
    # Prepare X and y
    X = data[numerical_features + categorical_features]
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✅ Train-Test Split:")
    print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Testing:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   Class balance: {y_train.value_counts(normalize=True).to_dict()}")
    
    # =========================================================================
    # PHASE 6: MODEL BUILDING & TRAINING
    # =========================================================================
    
    print_header("PHASE 6: MODEL BUILDING & TRAINING")
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    print("\n⏳ Applying preprocessing transformations...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"✅ Preprocessing complete:")
    print(f"   Input features:  {len(numerical_features) + len(categorical_features)}")
    print(f"   Output features: {X_train_processed.shape[1]}")
    
    # Initialize models
    print("\n📊 Initializing models...")
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=500),
            'reason': 'Linear baseline - fast, interpretable'
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'reason': 'Non-linear, captures interactions'
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_estimators=100),
            'reason': 'Ensemble - reduces overfitting'
        },
        'Support Vector Machine': {
            'model': SVC(probability=True, random_state=42),
            'reason': 'Kernel-based, high-dimensional'
        }
    }
    
    for name, info in models.items():
        print(f"\n✓ {name}")
        print(f"  → {info['reason']}")
    
    # Train models
    print("\n⏳ Training all models...")
    
    for name, info in models.items():
        print(f"\nTraining {name}...", end=" ")
        info['model'].fit(X_train_processed, y_train)
        print("✅")
    
    print("\n✅ All 4 models trained successfully")
    
    # =========================================================================
    # PHASE 7: MODEL EVALUATION
    # =========================================================================
    
    print_header("PHASE 7: MODEL EVALUATION")
    
    print_section("WHY ACCURACY ALONE IS NOT SUFFICIENT")
    print("""
In imbalanced datasets (Titanic: ~62% died, ~38% survived):

❌ PROBLEM: A model predicting "everyone dies" → 62% accuracy but USELESS!

✅ SOLUTION: Use multiple metrics:
   • PRECISION: Of predicted survivors, how many actually survived?
   • RECALL: Of actual survivors, how many did we identify?
   • F1-SCORE: Harmonic mean (best for imbalanced data)
""")
    
    # Evaluate each model
    results = {}
    
    for name, info in models.items():
        model = info['model']
        
        y_train_pred = model.predict(X_train_processed)
        y_test_pred = model.predict(X_test_processed)
        
        results[name] = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1])
        }
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train Acc': [f"{results[m]['train_accuracy']*100:.2f}%" for m in results],
        'Test Acc': [f"{results[m]['test_accuracy']*100:.2f}%" for m in results],
        'Precision': [f"{results[m]['precision']*100:.2f}%" for m in results],
        'Recall': [f"{results[m]['recall']*100:.2f}%" for m in results],
        'F1-Score': [f"{results[m]['f1_score']*100:.2f}%" for m in results]
    })
    
    # Sort by F1-Score (using numeric values)
    comparison_df['F1_numeric'] = [results[m]['f1_score'] for m in results]
    comparison_df = comparison_df.sort_values('F1_numeric', ascending=False)
    comparison_df = comparison_df.drop('F1_numeric', axis=1)
    
    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    print(comparison_df.to_string(index=False))
    
    # Print confusion matrices
    print("\n📋 CONFUSION MATRICES:")
    for name, metrics in results.items():
        cm = metrics['confusion_matrix']
        print(f"\n{name}:")
        print(f"                Predicted")
        print(f"                Died  Survived")
        print(f"Actual  Died    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"        Survived{cm[1,0]:4d}  {cm[1,1]:4d}")
        print(f"Accuracy: {metrics['test_accuracy']*100:.2f}% | F1: {metrics['f1_score']*100:.2f}%")
    
    # =========================================================================
    # PHASE 8: HYPERPARAMETER TUNING
    # =========================================================================
    
    print_header("PHASE 8: HYPERPARAMETER TUNING")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_base_score = results[best_model_name]['f1_score']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Base F1-Score: {best_base_score*100:.2f}%")
    
    print_section("WHY HYPERPARAMETER TUNING?")
    print("""
✓ Improves model accuracy (typically 2-5%)
✓ Better generalization to unseen data
✓ Finds optimal balance between bias and variance
✓ Systematic exploration vs. manual trial-and-error
""")
    
    # Define parameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        },
        'Decision Tree': {
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'criterion': ['gini', 'entropy']
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        }
    }
    
    param_grid = param_grids[best_model_name]
    
    print(f"\n⚙️  Tuning {best_model_name} with parameters:")
    for param, values in param_grid.items():
        print(f"   • {param}: {values}")
    
    print(f"\n⏳ Running GridSearchCV (5-fold cross-validation)...")
    
    grid_search = GridSearchCV(
        estimator=models[best_model_name]['model'],
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_processed, y_train)
    
    print(f"\n✅ Tuning complete!")
    
    print(f"\n🎯 Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   • {param}: {value}")
    
    improvement = (grid_search.best_score_ - best_base_score) * 100
    print(f"\n📈 Performance Improvement:")
    print(f"   Before: {best_base_score*100:.2f}%")
    print(f"   After:  {grid_search.best_score_*100:.2f}%")
    print(f"   Gain:   {'+' if improvement > 0 else ''}{improvement:.2f}%")
    
    best_tuned_model = grid_search.best_estimator_
    
    # =========================================================================
    # PHASE 9: PIPELINE CREATION
    # =========================================================================
    
    print_header("PHASE 9: PRODUCTION PIPELINE")
    
    print_section("WHY PIPELINES ARE CRITICAL FOR PRODUCTION")
    print("""
✓ Prevents data leakage (fit on train, transform on test)
✓ Ensures consistency (same preprocessing always)
✓ Simplifies deployment (one .pkl file = entire system)
✓ Enables testing (test entire pipeline as unit)
✓ Reduces errors (no manual preprocessing steps)
""")
    
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_tuned_model)
    ])
    
    print(f"\n✅ Pipeline created:")
    print(f"   Step 1: Preprocessor (ColumnTransformer)")
    print(f"           ├─ Numerical ({len(numerical_features)} features)")
    print(f"           └─ Categorical ({len(categorical_features)} features)")
    print(f"   Step 2: Classifier ({best_model_name})")
    
    # =========================================================================
    # PHASE 10: FINAL MODEL SELECTION & PREDICTION
    # =========================================================================
    
    print_header("PHASE 10: FINAL MODEL & PREDICTIONS")
    
    # Evaluate final model
    y_test_pred_final = best_tuned_model.predict(X_test_processed)
    
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred_final),
        'precision': precision_score(y_test, y_test_pred_final),
        'recall': recall_score(y_test, y_test_pred_final),
        'f1_score': f1_score(y_test, y_test_pred_final)
    }
    
    print(f"\n🏆 FINAL MODEL PERFORMANCE:")
    print(f"   Model:     {best_model_name}")
    print(f"   Accuracy:  {final_metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {final_metrics['precision']*100:.2f}%")
    print(f"   Recall:    {final_metrics['recall']*100:.2f}%")
    print(f"   F1-Score:  {final_metrics['f1_score']*100:.2f}%")
    
    # Example predictions
    print("\n📋 EXAMPLE PREDICTIONS:")
    
    examples = pd.DataFrame([
        {'pclass': 1, 'sex': 'female', 'age': 25, 'sibsp': 0, 'parch': 0,
         'fare': 100, 'embarked': 'S', 'family_size': 1, 'is_alone': 1,
         'desc': '1st class young woman'},
        {'pclass': 3, 'sex': 'male', 'age': 30, 'sibsp': 0, 'parch': 0,
         'fare': 7, 'embarked': 'Q', 'family_size': 1, 'is_alone': 1,
         'desc': '3rd class solo male'}
    ])
    
    for idx, row in examples.iterrows():
        desc = row['desc']
        passenger = row.drop('desc').to_frame().T
        
        pred = final_pipeline.predict(passenger)[0]
        proba = final_pipeline.predict_proba(passenger)[0]
        
        print(f"\n{idx+1}. {desc}")
        print(f"   → {'SURVIVED ✅' if pred == 1 else 'DID NOT SURVIVE ❌'}")
        print(f"   → Probability: {proba[1]*100:.1f}% survival")
    
    # =========================================================================
    # PHASE 11: MODEL SAVING
    # =========================================================================
    
    print_header("PHASE 11: MODEL PERSISTENCE")
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    model_path = '../models/titanic_production_pipeline.pkl'
    joblib.dump(final_pipeline, model_path)
    
    file_size = os.path.getsize(model_path) / 1024
    
    print(f"\n✅ Model saved successfully!")
    print(f"   Location: {model_path}")
    print(f"   Size:     {file_size:.2f} KB")
    
    print("\n💡 How to reuse:")
    print("   import joblib")
    print(f"   model = joblib.load('{model_path}')")
    print("   prediction = model.predict(new_data)")
    
    # =========================================================================
    # PHASE 12: USER INPUT & INFERENCE
    # =========================================================================
    
    print_header("PHASE 12: USER INPUT & INFERENCE")
    
    print("\n🎯 SIMULATING PRODUCTION INFERENCE:")
    
    # Load model (simulating production)
    production_model = joblib.load(model_path)
    
    # User input simulation
    user_passenger = pd.DataFrame([{
        'pclass': 2,
        'sex': 'female',
        'age': 35,
        'sibsp': 1,
        'parch': 2,
        'fare': 50,
        'embarked': 'C',
        'family_size': 4,
        'is_alone': 0
    }])
    
    print("\n📥 User Input:")
    print(user_passenger.to_dict('records')[0])
    
    prediction = production_model.predict(user_passenger)[0]
    probability = production_model.predict_proba(user_passenger)[0]
    
    print("\n📤 System Output:")
    if prediction == 1:
        print("   🟢 Result: SURVIVED")
    else:
        print("   🔴 Result: DID NOT SURVIVE")
    print(f"   📊 Confidence: {max(probability)*100:.1f}%")
    
    # =========================================================================
    # PHASE 13: DEPLOYMENT LAYER
    # =========================================================================
    
    print_header("PHASE 13: DEPLOYMENT OPTIONS")
    
    print("""
Two deployment options have been created:

1️⃣  STREAMLIT WEB APP (src/app_streamlit.py)
    • Best for: Demos, internal tools
    • Run: streamlit run src/app_streamlit.py
    • Features: User-friendly web interface
    
2️⃣  FASTAPI REST API (src/api_fastapi.py)
    • Best for: Production, integrations
    • Run: uvicorn src.api_fastapi:app --reload
    • Features: Scalable, documented API
    
See EXECUTION_GUIDE_PHASES_6_13.md for detailed deployment instructions.
""")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print_header("✅ PROJECT COMPLETE", "═")
    
    print(f"""
All Phases (6-13) Successfully Completed!

📊 FINAL RESULTS:
   • Best Model: {best_model_name}
   • Test Accuracy: {final_metrics['accuracy']*100:.2f}%
   • F1-Score: {final_metrics['f1_score']*100:.2f}%
   
📁 FILES CREATED:
   • {model_path}
   • src/app_streamlit.py (Web App)
   • src/api_fastapi.py (REST API)
   
🚀 NEXT STEPS:
   1. Test Streamlit app: streamlit run src/app_streamlit.py
   2. Test FastAPI: uvicorn src.api_fastapi:app --reload
   3. Review EXECUTION_GUIDE_PHASES_6_13.md
   
✨ This project is now:
   ✓ Production-ready
   ✓ Resume-ready
   ✓ Interview-ready
   ✓ GitHub-ready
""")
    
    print("═" * 80)
    print("Made with ❤️ by Senior ML Engineer | 2026-01-13")
    print("═" * 80)


if __name__ == "__main__":
    main()
