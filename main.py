import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page title and configuration
st.set_page_config(
    page_title="Concrete Compressive Strength Predictor",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Create a function to standardize probing type (same as in your notebook)
def standardize_probing_type(probing_type):
    if pd.isna(probing_type):
        return np.nan
    
    probing_type = str(probing_type).lower().strip()
    
    # Check for all variations of semi-direct
    if any(pattern in probing_type for pattern in ['semi-direct', 'semi direct', 'semidirect']):
        return 'Semi-Direct'
    elif 'semi' in probing_type and 'direct' in probing_type:
        return 'Semi-Direct'
    elif 'indirect' in probing_type:
        return 'Indirect'
    else:
        return 'Direct'  # Default to Direct for any other cases

# Create a function to preprocess the input data
def preprocess_input(input_data):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Extract grade value
    df['Grade'] = df['grade_of_concrete'].str.extract(r'(M-\d+)').fillna('Unknown')
    df['Grade_Value'] = df['Grade'].str.extract(r'M-(\d+)').astype(float)
    
    # Standardize probing type - apply to each element in the Series
    df['Type of Probing'] = df['type_of_probing'].apply(standardize_probing_type)
    
    # Map quality to score
    quality_map = {
        'Excellent': 5,
        'Good': 4,
        'Medium': 3,
        'Doubtful': 2,
        'Near Honey Combing': 1,
        'On Honey Combing': 0
    }
    df['Quality_Score'] = df['concrete_quality'].map(quality_map)
    
    # Create one-hot encoding for direction
    direction_dummies = pd.get_dummies(df['direction_of_impact'], prefix='Direction')
    df = pd.concat([df, direction_dummies], axis=1)
    
    # Create one-hot encoding for probing type
    probing_dummies = pd.get_dummies(df['Type of Probing'], prefix='Probing')
    df = pd.concat([df, probing_dummies], axis=1)
    
    # Convert numerical values
    df['Rebound_Number'] = pd.to_numeric(df['rebound_number'], errors='coerce')
    df['UPV'] = pd.to_numeric(df['pulse_velocity'], errors='coerce')
    
    # Select features for model input (same as in your notebook)
    features = ['Grade_Value', 'Rebound_Number', 'UPV', 'Quality_Score', 'pulse_velocity']
    
    # Add one-hot encoded columns
    features.extend([col for col in df.columns if col.startswith('Direction_')])
    features.extend([col for col in df.columns if col.startswith('Probing_')])
    
    # Check if all expected features are present, if not add them with zeros
    expected_direction_columns = ['Direction_Bottom', 'Direction_Horizontal', 
                                 'Direction_Middle', 'Direction_North', 
                                 'Direction_South', 'Direction_Top']
    
    expected_probing_columns = ['Probing_Direct', 'Probing_Indirect', 'Probing_Semi-Direct']
    
    for col in expected_direction_columns:
        if col not in df.columns:
            df[col] = 0
            features.append(col)
    
    for col in expected_probing_columns:
        if col not in df.columns:
            df[col] = 0
            features.append(col)
    
    # Create feature matrix
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

# Create the Streamlit UI
st.title("Concrete Compressive Strength Predictor")
st.write("Enter the concrete properties to predict its compressive strength")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    grade_of_concrete = st.text_input("Grade of Concrete (e.g., M-35)", "M-35")
    rebound_number = st.number_input("Average Rebound Number", min_value=0.0, max_value=100.0, value=40.0)
    pulse_velocity = st.number_input("Pulse Velocity (km/sec)", min_value=0.0, max_value=10.0, value=4.0)
    
with col2:
    direction_options = ["Horizontal", "Vertical", "Top", "Bottom", "North", "South", "Middle"]
    direction_of_impact = st.selectbox("Direction of Impact", direction_options)
    
    probing_options = ["Direct", "Indirect", "Semi-Direct"]
    type_of_probing = st.selectbox("Type of Probing", probing_options)
    
    quality_options = ["Excellent", "Good", "Medium", "Doubtful", "Near Honey Combing", "On Honey Combing"]
    concrete_quality = st.selectbox("Concrete Quality Grading", quality_options)

# Create a button to make prediction
if st.button("Predict Compressive Strength"):
    # Collect input data
    input_data = {
        'grade_of_concrete': grade_of_concrete,
        'rebound_number': rebound_number,
        'pulse_velocity': pulse_velocity,
        'direction_of_impact': direction_of_impact,
        'type_of_probing': type_of_probing,
        'concrete_quality': concrete_quality
    }
    
    # Preprocess the input
    X_processed = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(X_processed)
    
    # Display the result
    st.success(f"The predicted compressive strength is: {prediction[0]:.2f} N/mm²")
    
    # Visualization of the result
    st.subheader("Prediction Visualization")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['Predicted Strength'], [prediction[0]], color='blue', width=0.4)
    ax.set_ylabel('Compressive Strength (N/mm²)')
    ax.set_ylim(0, max(50, prediction[0] * 1.2))  # Set y-limit with some headroom
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate([prediction[0]]):
        ax.text(i, v + 0.5, f"{v:.2f}", ha='center', fontweight='bold')
    
    st.pyplot(fig)

# Add information about the model
with st.expander("About this Model"):
    st.write("""
    This application uses an XGBoost regression model trained on concrete compressive strength data.
    The model takes various concrete properties as input and predicts the compressive strength in N/mm².
    
    **Features used in the model:**
    - Grade of Concrete (M-xx)
    - Rebound Number
    - Pulse Velocity (UPV)
    - Direction of Impact
    - Type of Probing
    - Concrete Quality
    
    The model was trained on a dataset of concrete samples with known compressive strengths.
    """)
