import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. إعداد الصفحة
st.set_page_config(page_title="Telco Churn AI", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stButton>button {width: 100%; background-color: #FF4B4B; color: white;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# 2. تحميل الموديل والسكيلر (ضروري جداً)
try:
    model = joblib.load('churn_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')  # تحميل السكيلر
except FileNotFoundError:
    st.error("⚠️ ملفات الموديل ناقصة! تأكد من وجود 'churn_model.pkl' و 'scaler.pkl'")
    st.stop()

# 3. العنوان
st.title("📊 Customer Churn Prediction AI")
st.markdown("---")

# 4. واجهة الإدخال
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.radio("Senior Citizen?", ["No", "Yes"], horizontal=True)
    partner = st.radio("Has Partner?", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Has Dependents?", ["No", "Yes"], horizontal=True)
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.subheader("💳 Account Details")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.radio("Paperless Billing?", ["No", "Yes"], horizontal=True)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1500.0)

with col3:
    st.subheader("📡 Services")
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    phone_service = st.radio("Phone Service?", ["Yes", "No"], horizontal=True)
    multiple_lines = st.radio("Multiple Lines?", ["Yes", "No"], horizontal=True) if phone_service == "Yes" else "No phone service"
    
    with st.expander("➕ Additional Services"):
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# 5. التوقع
st.markdown("---")
if st.button("🚀 Predict Result"):
    
    # قائمة الأعمدة (لازم تكون نفس ترتيب التدريب بالظبط)
    columns = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
        'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
        'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 
        'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
        'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 
        'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
        'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    input_df = pd.DataFrame(0, index=[0], columns=columns)
    
    # --- تعبئة البيانات ---
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['TotalCharges'] = total_charges
    input_df['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    
    if gender == "Male": input_df['gender_Male'] = 1
    if partner == "Yes": input_df['Partner_Yes'] = 1
    if dependents == "Yes": input_df['Dependents_Yes'] = 1
    if phone_service == "Yes": input_df['PhoneService_Yes'] = 1
    if paperless == "Yes": input_df['PaperlessBilling_Yes'] = 1
    
    if multiple_lines == "Yes": input_df['MultipleLines_Yes'] = 1
    elif multiple_lines == "No phone service": input_df['MultipleLines_No phone service'] = 1
        
    if internet_service == "Fiber optic": input_df['InternetService_Fiber optic'] = 1
    elif internet_service == "No": input_df['InternetService_No'] = 1
        
    if online_security == "Yes": input_df['OnlineSecurity_Yes'] = 1
    elif online_security == "No internet service": input_df['OnlineSecurity_No internet service'] = 1

    if tech_support == "Yes": input_df['TechSupport_Yes'] = 1
    elif tech_support == "No internet service": input_df['TechSupport_No internet service'] = 1
    
    if online_backup == "Yes": input_df['OnlineBackup_Yes'] = 1
    elif online_backup == "No internet service": input_df['OnlineBackup_No internet service'] = 1
        
    if device_protection == "Yes": input_df['DeviceProtection_Yes'] = 1
    elif device_protection == "No internet service": input_df['DeviceProtection_No internet service'] = 1

    if streaming_tv == "Yes": input_df['StreamingTV_Yes'] = 1
    elif streaming_tv == "No internet service": input_df['StreamingTV_No internet service'] = 1
        
    if streaming_movies == "Yes": input_df['StreamingMovies_Yes'] = 1
    elif streaming_movies == "No internet service": input_df['StreamingMovies_No internet service'] = 1
    
    if contract == "One year": input_df['Contract_One year'] = 1
    elif contract == "Two year": input_df['Contract_Two year'] = 1
        
    if payment == "Credit card (automatic)": input_df['PaymentMethod_Credit card (automatic)'] = 1
    elif payment == "Electronic check": input_df['PaymentMethod_Electronic check'] = 1
    elif payment == "Mailed check": input_df['PaymentMethod_Mailed check'] = 1

    # ==========================================
    # ⚠️ الخطوة الحاسمة: تطبيق الـ Scaling
    # ==========================================
    # لازم نختار نفس الأعمدة اللي عملنا عليها fit في النوت بوك
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # ==========================================
    # ⚠️ تصحيح خطأ الـ Scaling
    # ==========================================
    
    # بما إن السكيلر اتدرب على الداتا كلها، لازم نبعت له الـ DataFrame كله
    try:
        # السطر ده هيحول الداتا كلها (الـ 30 عمود) بناءً على اللي اتعلمه
        input_df = scaler.transform(input_df)
    except Exception as e:
        st.error(f"خطأ في الـ Scaling: {e}")
        st.stop()

    # --- التوقع ---
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # عرض النتائج
    r1, r2 = st.columns([1, 2])
    with r1:
        if prediction == 1:
            st.error("⚠️ CHURN")
            st.metric("Risk Probability", f"{probability*100:.1f}%", "High", delta_color="inverse")
        else:
            st.success("✅ SAFE")
            st.metric("Risk Probability", f"{probability*100:.1f}%", "Low")
            
    with r2:
        st.write("Risk Meter:")
        st.progress(int(probability * 100))