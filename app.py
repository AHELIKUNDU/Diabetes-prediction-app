import streamlit as st
import joblib
import numpy as np
import time
import pandas as pd

# Load the model
model = joblib.load("diabetes_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stNumberInput > div > div > input {
        font-size: 16px;
    }
    .css-1r6slb0 {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .result-success {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .result-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .header-title {
        background: linear-gradient(120deg, #2c3e50, #3498db);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        text-align: center;
    }
    /* Enhanced How-to section styling */
    .how-to-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 20px 0 30px 0;
        border: 3px solid #ffffff;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .how-to-section h4 {
        font-size: 28px;
        margin-bottom: 15px;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .how-to-section ol {
        font-size: 18px;
        line-height: 2;
        padding-left: 25px;
        color: #ffffff;
    }
    .how-to-section ol li {
        margin-bottom: 5px;
        font-weight: 500;
    }
    .how-to-section .warning-text {
        background: rgba(255, 255, 255, 0.2);
        padding: 12px;
        border-radius: 10px;
        margin-top: 15px;
        border-left: 4px solid #ffd700;
        font-size: 16px;
        color: #ffffff;
    }
    .how-to-section .warning-text strong {
        color: #ffd700;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-title">
        <h1>🩺 Diabetes Risk Prediction</h1>
        <p style="font-size: 18px; opacity: 0.9;">Enter patient details to assess diabetes risk using machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Enhanced visible How-to section
st.markdown("""
    <div class="how-to-section">
        <h4>📋 How to Use This App</h4>
        <ol>
            <li><strong>Step 1:</strong> Enter all patient health metrics in the input fields below</li>
            <li><strong>Step 2:</strong> Click the <strong style="color: #ffd700;">"🔍 Predict Diabetes Risk"</strong> button</li>
            <li><strong>Step 3:</strong> View the prediction results and personalized recommendations</li>
        </ol>
        <div class="warning-text">
            <strong>⚠️ Important Note:</strong> This is a screening tool based on machine learning and is not a substitute for professional medical advice. 
            Always consult with a healthcare provider for accurate diagnosis.
        </div>
    </div>
""", unsafe_allow_html=True)

# Create two columns for input layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Patient Information")
    st.markdown("---")
    
    pregnancies = st.number_input(
        "🤰 Pregnancies", 
        min_value=0, 
        max_value=20,
        help="Number of times pregnant"
    )
    
    glucose = st.number_input(
        "🍬 Glucose", 
        min_value=0, 
        max_value=300,
        help="Plasma glucose concentration"
    )
    
    blood_pressure = st.number_input(
        "💓 Blood Pressure", 
        min_value=0, 
        max_value=200,
        help="Diastolic blood pressure (mm Hg)"
    )
    
    skin_thickness = st.number_input(
        "📏 Skin Thickness", 
        min_value=0, 
        max_value=100,
        help="Triceps skin fold thickness (mm)"
    )

with col2:
    st.markdown("### 📊 Health Metrics")
    st.markdown("---")
    
    insulin = st.number_input(
        "💉 Insulin", 
        min_value=0, 
        max_value=900,
        help="2-Hour serum insulin (mu U/ml)"
    )
    
    bmi = st.number_input(
        "⚖️ BMI", 
        min_value=0.0, 
        max_value=70.0,
        step=0.1,
        help="Body Mass Index (weight in kg/(height in m)^2)"
    )
    
    diabetes_pedigree = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=2.5,
        step=0.01,
        help="Diabetes pedigree function (family history)"
    )
    
    age = st.number_input(
        "🎂 Age", 
        min_value=1, 
        max_value=120,
        help="Age in years"
    )

# Add a separator
st.markdown("---")

# Prediction button with enhanced styling
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔍 Predict Diabetes Risk", use_container_width=True)

# Display prediction results
if predict_button:
    # Create input array
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])
    
    # Show loading spinner
    with st.spinner('Analyzing patient data...'):
        time.sleep(1.5)  # Simulate processing
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
    
    # Display results in a nice format
    st.markdown("### 📊 Prediction Results")
    st.markdown("---")
    
    # Create columns for results
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
    
    with result_col2:
        if prediction == 1:
            st.markdown("""
                <div class="result-box result-danger">
                    ⚠️ Diabetes Risk Detected
                </div>
            """, unsafe_allow_html=True)
            
            # Show risk probability
            st.info(f"**Risk Probability:** {probabilities[1]*100:.1f}%")
            st.warning("""
                **Recommendations:**
                - Consult a healthcare professional
                - Monitor blood sugar levels regularly
                - Consider lifestyle changes
                - Regular exercise and healthy diet
            """)
            
        else:
            st.markdown("""
                <div class="result-box result-success">
                    ✅ No Diabetes Risk Detected
                </div>
            """, unsafe_allow_html=True)
            
            # Show risk probability
            st.success(f"**Confidence:** {probabilities[0]*100:.1f}%")
            st.info("""
                **Recommendations:**
                - Maintain a healthy lifestyle
                - Regular health check-ups
                - Balanced diet and exercise
                - Stay hydrated
            """)
    
    # Show all inputs in a neat table
    st.markdown("### 📋 Patient Data Summary")
    st.markdown("---")
    
    # Create a summary dataframe
    summary_data = {
        "Parameter": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", 
                     "Insulin", "BMI", "Diabetes Pedigree", "Age"],
        "Value": [pregnancies, glucose, blood_pressure, skin_thickness, 
                 insulin, bmi, diabetes_pedigree, age],
        "Normal Range": ["0-20", "70-140", "60-90", "10-40", 
                        "2-20", "18.5-24.9", "0.0-0.5", "20-65"]
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Made with ❤️ using Streamlit | For educational purposes only</p>
    </div>
""", unsafe_allow_html=True)