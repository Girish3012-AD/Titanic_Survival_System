"""
Titanic Survival Prediction - Inference Script
Author: ML Engineer
Description: Load trained model and make predictions on new passenger data
"""

import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Union


class TitanicSurvivalPredictor:
    """
    A class to handle Titanic survival predictions using a trained model.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor by loading the trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        if model_path is None:
            # Get the path to the model relative to this script
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'titanic_survival_model.pkl')
            
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    
    def preprocess_input(self, passenger_data: Dict) -> pd.DataFrame:
        """
        Preprocess input data to match training format.
        
        Args:
            passenger_data (dict): Dictionary containing passenger information
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        # Create DataFrame
        df = pd.DataFrame([passenger_data])
        
        # Add engineered features if not present
        if 'family_size' not in df.columns:
            df['family_size'] = df['sibsp'] + df['parch'] + 1
        
        if 'is_alone' not in df.columns:
            df['is_alone'] = (df['family_size'] == 1).astype(int)
        
        return df
    
    def predict(self, passenger_data: Dict) -> Dict[str, Union[int, float]]:
        """
        Predict survival for a passenger.
        
        Args:
            passenger_data (dict): Passenger information
            
        Returns:
            dict: Prediction results with survival status and probability
        """
        # Preprocess
        df = self.preprocess_input(passenger_data)
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        return {
            'survived': int(prediction),
            'survival_probability': float(probability[1]),
            'death_probability': float(probability[0])
        }
    
    def predict_batch(self, passengers_data: list) -> pd.DataFrame:
        """
        Predict survival for multiple passengers.
        
        Args:
            passengers_data (list): List of passenger dictionaries
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        results = []
        for passenger in passengers_data:
            result = self.predict(passenger)
            results.append(result)
        
        return pd.DataFrame(results)


def main():
    """
    Example usage of the TitanicSurvivalPredictor class.
    """
    print("\n" + "="*70)
    print("🚢 TITANIC SURVIVAL PREDICTION SYSTEM")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = TitanicSurvivalPredictor()
    
    
    # Example passengers
    passengers = [
        {
            'pclass': 1,
            'sex': 'female',
            'age': 25,
            'sibsp': 0,
            'parch': 0,
            'fare': 100,
            'embarked': 'S'
        },
        {
            'pclass': 3,
            'sex': 'male',
            'age': 30,
            'sibsp': 1,
            'parch': 2,
            'fare': 15,
            'embarked': 'S'
        },
        {
            'pclass': 2,
            'sex': 'female',
            'age': 18,
            'sibsp': 0,
            'parch': 1,
            'fare': 30,
            'embarked': 'C'
        }
    ]
    
    # Make predictions
    print("📊 PREDICTIONS:\n")
    for i, passenger in enumerate(passengers, 1):
        result = predictor.predict(passenger)
        
        print(f"Passenger {i}:")
        print(f"  Class: {passenger['pclass']}, Sex: {passenger['sex']}, Age: {passenger['age']}")
        print(f"  Family: {passenger['sibsp']} siblings/spouse, {passenger['parch']} parents/children")
        print(f"  Fare: ${passenger['fare']}")
        print(f"  🔮 Prediction: {'SURVIVED ✅' if result['survived'] else 'DID NOT SURVIVE ❌'}")
        print(f"  📈 Survival Probability: {result['survival_probability']:.2%}")
        print()
    
    print("="*70)


if __name__ == "__main__":
    main()
