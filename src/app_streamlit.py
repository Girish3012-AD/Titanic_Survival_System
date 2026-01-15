"""
Titanic Survival Prediction - Streamlit Web Application
Author: Senior ML Engineer
Phase 13: Deployment Layer - Option 1

This is a production-ready Streamlit app for predicting Titanic passenger survival.

Run with: streamlit run src/app_streamlit.py
"""

import streamlit as st
import joblib
import pandas as pd
import os
import sys

# Configure page
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .survived {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .not-survived {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    model_path = 'models/titanic_production_pipeline.pkl'
    
    # Try alternate paths if not found
    if not os.path.exists(model_path):
        model_path = '../models/titanic_production_pipeline.pkl'
    if not os.path.exists(model_path):
        model_path = 'models/titanic_survival_model.pkl'
    if not os.path.exists(model_path):
        model_path = '../models/titanic_survival_model.pkl'
    
    try:
        model = joblib.load(model_path)
        return model, model_path
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error(f"Please ensure the model file exists at: {model_path}")
        return None, None


def create_input_features():
    """Create input form for passenger data."""
    
    st.sidebar.header("📋 Passenger Information")
    st.sidebar.write("Enter passenger details to predict survival:")
    
    # Passenger Class
    pclass = st.sidebar.selectbox(
        "🎫 Passenger Class",
        options=[1, 2, 3],
        help="1 = First Class, 2 = Second Class, 3 = Third Class"
    )
    
    # Gender
    sex = st.sidebar.radio(
        "👤 Gender",
        options=["male", "female"],
        horizontal=True
    )
    
    # Age
    age = st.sidebar.slider(
        "🎂 Age",
        min_value=0,
        max_value=80,
        value=25,
        help="Age in years"
    )
    
    # Family aboard
    st.sidebar.subheader("👨‍👩‍👧‍👦 Family Aboard")
    
    sibsp = st.sidebar.number_input(
        "Siblings/Spouses",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of siblings or spouses aboard"
    )
    
    parch = st.sidebar.number_input(
        "Parents/Children",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of parents or children aboard"
    )
    
    # Fare
    fare = st.sidebar.number_input(
        "💰 Fare (in £)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=0.5,
        help="Ticket fare in British Pounds"
    )
    
    # Port of Embarkation
    embarked = st.sidebar.selectbox(
        "⚓ Port of Embarkation",
        options=["S", "C", "Q"],
        help="S = Southampton, C = Cherbourg, Q = Queenstown"
    )
    
    # Calculate derived features
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    return {
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked,
        'family_size': family_size,
        'is_alone': is_alone
    }


def display_passenger_summary(passenger_data):
    """Display passenger information summary."""
    
    st.subheader("📊 Passenger Profile Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Class", passenger_data['pclass'])
        st.metric("Gender", passenger_data['sex'].title())
    
    with col2:
        st.metric("Age", f"{passenger_data['age']} years")
        st.metric("Fare", f"£{passenger_data['fare']:.2f}")
    
    with col3:
        port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
        st.metric("Embarked", port_names[passenger_data['embarked']])
        st.metric("Family Size", passenger_data['family_size'])
    
    with col4:
        st.metric("Siblings/Spouses", passenger_data['sibsp'])
        st.metric("Parents/Children", passenger_data['parch'])


def make_prediction(model, passenger_data):
    """Make survival prediction."""
    
    # Create DataFrame (remove derived features if model doesn't need them)
    df = pd.DataFrame([{
        'pclass': passenger_data['pclass'],
        'sex': passenger_data['sex'],
        'age': passenger_data['age'],
        'sibsp': passenger_data['sibsp'],
        'parch': passenger_data['parch'],
        'fare': passenger_data['fare'],
        'embarked': passenger_data['embarked'],
        'family_size': passenger_data['family_size'],
        'is_alone': passenger_data['is_alone']
    }])
    
    try:
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        return prediction, probability
    except Exception as e:
        # Some models might not have all features
        # Try without engineered features
        df = pd.DataFrame([{
            'pclass': passenger_data['pclass'],
            'sex': passenger_data['sex'],
            'age': passenger_data['age'],
            'sibsp': passenger_data['sibsp'],
            'parch': passenger_data['parch'],
            'fare': passenger_data['fare'],
            'embarked': passenger_data['embarked']
        }])
        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        return prediction, probability


def display_prediction(prediction, probability):
    """Display prediction results."""
    
    st.subheader("🔮 Prediction Results")
    
    survival_prob = probability[1] * 100
    death_prob = probability[0] * 100
    
    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-box survived">
                ✅ PASSENGER WOULD HAVE SURVIVED
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box not-survived">
                ❌ PASSENGER WOULD NOT HAVE SURVIVED
            </div>
        """, unsafe_allow_html=True)
    
    # Probability bars
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🟢 Survival Probability", f"{survival_prob:.1f}%")
        st.progress(survival_prob / 100)
    
    with col2:
        st.metric("🔴 Death Probability", f"{death_prob:.1f}%")
        st.progress(death_prob / 100)
    
    # Confidence interpretation
    st.subheader("📊 Confidence Level")
    
    max_prob = max(survival_prob, death_prob)
    
    if max_prob >= 80:
        confidence = "Very High"
        color = "green"
    elif max_prob >= 65:
        confidence = "High"
        color = "blue"
    elif max_prob >= 55:
        confidence = "Moderate"
        color = "orange"
    else:
        confidence = "Low"
        color = "red"
    
    st.markdown(f"**Confidence:** :{color}[{confidence}] ({max_prob:.1f}%)")


def main():
    """Main application."""
    
    # Header
    st.title("🚢 Titanic Survival Prediction System")
    st.markdown("---")
    st.markdown("""
        Welcome to the **Titanic Survival Predictor**! This machine learning application
        predicts whether a passenger would have survived the Titanic disaster based on
        their demographic and travel information.
        
        **Model:** Production-ready Random Forest classifier with 82%+ accuracy
    """)
    
    # Load model
    model, model_path = load_model()
    
    if model is None:
        st.stop()
    
    st.success(f"✅ Model loaded from: `{model_path}`")
    st.markdown("---")
    
    # Get user input
    passenger_data = create_input_features()
    
    # Display passenger summary
    display_passenger_summary(passenger_data)
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Survival", type="primary"):
        with st.spinner("Making prediction..."):
            prediction, probability = make_prediction(model, passenger_data)
            display_prediction(prediction, probability)
        
        # Historical context
        st.markdown("---")
        st.subheader("📚 Historical Context")
        st.info("""
            **Titanic Facts:**
            - The RMS Titanic sank on April 15, 1912
            - Of 2,224 passengers and crew, approximately 1,500 died
            - Survival rates: Women 74%, Men 19%
            - First class passengers had 63% survival rate
            - Third class passengers had 24% survival rate
        """)
    
    # About section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
            ### Model Details
            - **Algorithm:** Random Forest Classifier (Tuned)
            - **Accuracy:** 82%+
            - **Features:** Passenger class, gender, age, family size, fare, embarkation port
            
            ### Tech Stack
            - **Framework:** Streamlit
            - **ML Library:** Scikit-Learn
            - **Model:** Joblib serialized pipeline
            
            ### Developer
            Built as part of an end-to-end ML engineering project following
            industry best practices for production deployment.
        """)


if __name__ == "__main__":
    main()
