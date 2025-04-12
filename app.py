import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Credit Scoring Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stSelectbox, .stNumberInput {background-color: #4CAF50; border-radius: 5px; padding: 5px;}
    .sidebar .sidebar-content {background-color: #e0e0e0;}
    h1 {color: #2c3e50; font-family: 'Arial';}
    h2 {color: #34495e; font-family: 'Arial';}
    </style>
""", unsafe_allow_html=True)

# Load trained model
model = joblib.load('credit_model (1).pkl')

# Load scaler if available, otherwise use placeholder
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.warning("Scaler file 'scaler.pkl' not found. Using a new scaler, which may affect prediction accuracy.")
    scaler = StandardScaler()

# Define mappings
categorical_mappings = {
    'Status of existing checking account': {
        '< 0 DM': 0, '0 to 200 DM': 1, '>= 200 DM or salary for 1+ year': 2, 'No checking account': 3
    },
    'Credit history': {
        'No credits taken': 0, 'All credits paid back duly at this bank': 1, 
        'Existing credits paid back duly': 2, 'Delay in paying off in the past': 3, 
        'Critical account/other credits elsewhere': 4
    },
    'Savings account/bonds': {
        '< 100 DM': 0, '100 to 500 DM': 1, '500 to 1000 DM': 2, '>= 1000 DM': 3, 
        'Unknown/no savings': 4
    },
    'Present employment since': {
        'Unemployed': 0, '< 1 year': 1, '1 to 4 years': 2, '4 to 7 years': 3, 
        '>= 7 years': 4
    },
    'Property': {
        'Real estate': 0, 'Building society savings/life insurance': 1, 
        'Car or other': 2, 'Unknown/no property': 3
    },
    'Other installment plans ': {
        'Bank': 0, 'Stores': 1, 'None': 2
    }
}

# Model features in exact training order
expected_features = [
    'Status of existing checking account', 'Credit history', 'Duration in month',
    'Savings account/bonds', 'Credit amount', 'Property', 'Debt_to_Income',
    'Present employment since', 'Other installment plans ', 'Age in years'
]

numerical_features = ['Duration in month', 'Credit amount', 'Debt_to_Income', 'Age in years']

# Sidebar inputs
with st.sidebar:
    st.header("Input Your Details")
    
    with st.expander("Personal and Financial Status", expanded=True):
        checking_account = st.selectbox("Checking Account Status", list(categorical_mappings['Status of existing checking account'].keys()))
        credit_history = st.selectbox("Credit History", list(categorical_mappings['Credit history'].keys()))
        savings = st.selectbox("Savings Account/Bonds", list(categorical_mappings['Savings account/bonds'].keys()))
        employment = st.selectbox("Employment Duration", list(categorical_mappings['Present employment since'].keys()))
    
    with st.expander("Assets and Obligations", expanded=True):
        property = st.selectbox("Property Owned", list(categorical_mappings['Property'].keys()))
        installment_plans = st.selectbox("Other Installment Plans", list(categorical_mappings['Other installment plans '].keys()))
    
    with st.expander("Financial Details", expanded=True):
        duration = st.number_input("Loan Duration (months)", min_value=1, max_value=72, value=12)
        credit_amount = st.number_input("Credit Amount (DM)", min_value=100, max_value=20000, value=1000)
        age = st.number_input("Age (years)", min_value=19, max_value=75, value=30)
        installment_rate = st.number_input("Installment Rate (% of income)", min_value=1, max_value=4, value=2)
    
    predict_button = st.button("Predict Creditworthiness")

# Main panel
st.title("Credit Scoring Dashboard")
st.write("Analyze your creditworthiness with interactive insights.")

if predict_button:
    # Calculate Debt_to_Income
    debt_to_income = credit_amount * installment_rate / 100

    # Create input_data as a list in exact training order
    input_values = [
        categorical_mappings['Status of existing checking account'][checking_account],
        categorical_mappings['Credit history'][credit_history],
        duration,
        categorical_mappings['Savings account/bonds'][savings],
        credit_amount,
        categorical_mappings['Property'][property],
        debt_to_income,
        categorical_mappings['Present employment since'][employment],
        categorical_mappings['Other installment plans '][installment_plans],
        age
    ]

    # Convert to DataFrame for scaling
    input_data = pd.DataFrame([input_values], columns=expected_features)

    # Scale numerical features
    scaled_values = scaler.transform(input_data[numerical_features])
    input_data[numerical_features] = scaled_values

    # Convert to NumPy array in exact order for prediction
    input_array = input_data[expected_features].to_numpy()

    # Prediction
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)[0]
    result = "Good" if prediction[0] == 1 else "Bad"

    # Layout for results and visualizations
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Prediction Result")
        st.success(f"Creditworthiness: **{result}**")
        st.write("**Good**: Likely to receive credit. **Bad**: Higher risk.")
        
        # Credit Score Meter
        st.subheader("Credit Score Meter")
        prob_good = prediction_proba[1] * 100  # Probability of "Good" in percentage

        # Map probability to credit score categories (0-100%)
        if prob_good < 40:
            category = "Poor"
            color = '#ff4d4d'  # Red
        elif 40 <= prob_good < 60:
            category = "Fair"
            color = '#FFA500'  # orange
        elif 60 <= prob_good < 80:
            category = "Good"
            color = '#008000'  # green
        else:
            category = "Excellent"
            color = '#006600'  # Dark Green

        # Create the meter
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location('W')  # Start from the left (west)
        ax.set_theta_direction(-1)       # Move clockwise

        # Draw background arcs (0-100%, 180 degrees)
        angles = np.linspace(0, np.pi, 100)
        r = np.ones_like(angles) * 1.0
        ax.plot(angles, r, color='lightgrey', linewidth=30, solid_capstyle='butt')

        # Draw "filled" portion based on probability (0-100% maps to 0-180 degrees)
        theta = (prob_good / 100) * np.pi
        filled_angle = np.linspace(0, theta, 50)
        ax.plot(filled_angle, np.ones_like(filled_angle) * 1.0, color=color, linewidth=30, solid_capstyle='butt')

        # Remove polar extras
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)

        # Add labels for categories
        ax.text(np.pi/4, 1.2, "Poor", color='red', ha='center', va='center', fontsize=10)
        ax.text(3*np.pi/4, 1.2, "Fair", color='orange', ha='center', va='center', fontsize=10)
        ax.text(5*np.pi/4, 1.2, "Good", color='green', ha='center', va='center', fontsize=10)
        ax.text(7*np.pi/4, 1.2, "Excellent", color='darkgreen', ha='center', va='center', fontsize=10)

        # Add probability text in the center
        ax.text(0, 0, f"{prob_good:.1f}%\n{category}", ha='center', va='center', fontsize=14, fontweight='bold', color='black')

        st.pyplot(fig)

    with col2:
        st.subheader("Input Summary")
        input_display = pd.DataFrame(columns=expected_features, index=['Value'])
        for col in expected_features:
            if col in categorical_mappings:
                # Map numerical value back to human-readable label
                numerical_value = input_data[col].iloc[0]
                reverse_map = {v: k for k, v in categorical_mappings[col].items()}
                human_readable = reverse_map[numerical_value]
                input_display[col] = human_readable
            else:
                # For numerical features, format appropriately
                if col in numerical_features:
                    value = input_data[col].iloc[0]
                    if col == 'Debt_to_Income':
                        input_display[col] = f"{value:.2f}"
                    else:
                        input_display[col] = value
        st.dataframe(input_display, height=300)

        # Feature Importance
        st.subheader("Model Feature Importance")
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': expected_features,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
        ax.set_title("Feature Importance in Prediction")
        st.pyplot(fig)

else:
    st.info("Enter your details in the sidebar and click 'Predict Creditworthiness' to see results.")
    img_url = "https://img.freepik.com/free-vector/modern-credit-score-scale-meter-concept-design_1017-53354.jpg"

    st.markdown(
    f"""
    <div style="text-align: center;">
    <img src="{img_url}" alt="Credit Score Meter" style="height:400px;">
    </div>
    """,
    unsafe_allow_html=True
    )