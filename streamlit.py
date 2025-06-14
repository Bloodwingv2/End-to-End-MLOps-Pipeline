import streamlit as st
import requests
import json

# FastAPI backend URL
API_BASE_URL = "https://mlops-1tdv.onrender.com"

# Ping FastApi Backend to wakeup for inference
try:
    requests.get(API_BASE_URL, timeout=5)
except:
    st.warning("Backend still waking up...")

# Streamlit UI
st.title("🧬 Disease Prediction ML Pipeline")
st.sidebar.title("🔍 Select a Disease")

disease_option = st.sidebar.selectbox(
    "Choose a Disease:",
    ["Diabetes", "Heart Disease", "Parkinson's"]
)

# Preset values (same as before)
healthy_man = {
    "pregnancies": 1, "glucose": 85, "blood_pressure": 70, "skin_thickness": 20,
    "insulin": 80, "bmi": 24.0, "dpf": 0.5, "age": 30
}

diabetic_man = {
    "pregnancies": 5, "glucose": 150, "blood_pressure": 85, "skin_thickness": 35,
    "insulin": 200, "bmi": 32.0, "dpf": 1.2, "age": 50
}

healthy_heart = {
    "age": 30, "sex": 1, "cp": 0, "trestbps": 120, "chol": 180, "fbs": 0,
    "restecg": 0, "thalach": 170, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
}

unhealthy_heart = {
    "age": 65, "sex": 1, "cp": 2, "trestbps": 150, "chol": 280, "fbs": 1,
    "restecg": 1, "thalach": 120, "exang": 1, "oldpeak": 2.5, "slope": 1, "ca": 2, "thal": 1
}

healthy_parkinsons = {
    "fo": 200.0, "fhi": 220.0, "flo": 180.0, "jitter": 0.002, "jitter_abs": 0.00002,
    "rap": 0.002, "ppq": 0.003, "ddp": 0.006, "shimmer": 0.01, "shimmer_db": 0.1,
    "apq3": 0.005, "apq5": 0.007, "apq": 0.01, "dda": 0.01, "nhr": 0.01,
    "hnr": 30.0, "rpde": 0.4, "dfa": 0.6, "spread1": -5.0, "spread2": 0.1, "d2": 2.0, "ppe": 0.05
}

unhealthy_parkinsons = {
    "fo": 150.0, "fhi": 200.0, "flo": 100.0, "jitter": 0.005, "jitter_abs": 0.0005,
    "rap": 0.03, "ppq": 0.02, "ddp": 0.02, "shimmer": 0.1, "shimmer_db": 0.05,
    "apq3": 0.02, "apq5": 0.03, "apq": 0.02, "dda": 0.01, "nhr": 0.02,
    "hnr": 30.0, "rpde": 0.4, "dfa": 0.6, "spread1": -5.0, "spread2": 0.2, "d2": 2.0, "ppe": 0.1
}

# Initialize session state
if "inputs" not in st.session_state:
    st.session_state.inputs = healthy_man.copy()
if "inputs2" not in st.session_state:
    st.session_state.inputs2 = healthy_heart.copy()
if "inputs3" not in st.session_state:
    st.session_state.inputs3 = healthy_parkinsons.copy()

# Helper function to call API
def call_api(endpoint, data):
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"❌ Error calling API: {str(e)}")
        return None

# Helper function to get metrics from API
def get_metrics(disease):
    try:
        response = requests.get(f"{API_BASE_URL}/metrics/{disease}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Check API health
try:
    health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("models_loaded"):
            st.sidebar.success("🟢 API Connected")
        else:
            st.sidebar.warning("🟡 API Connected but models not loaded")
    else:
        st.sidebar.error("🔴 API Health Check Failed")
except:
    st.sidebar.error("🔴 API Disconnected")

if disease_option == "Diabetes":
    st.subheader("Diabetes Prediction")
    
    # Get metrics from API
    metrics = get_metrics("diabetes")
    if metrics:
        st.markdown(
            f"**🧪 Accuracy:** {metrics['accuracy'] * 100:.2f}% | "
            f"**🎯 Precision:** {metrics['precision'] * 100:.2f}% | "
            f"**📡 Recall:** {metrics['recall'] * 100:.2f}% | "
            f"**📊 F1 Score:** {metrics['f1_score'] * 100:.2f}%"
        )
    
    # Preset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Fill Healthy Values"):
            st.session_state.inputs = healthy_man.copy()
            st.rerun()
    with col2:
        if st.button("⚠️ Fill Diabetic Values"):
            st.session_state.inputs = diabetic_man.copy()
            st.rerun()
    
    # Input fields
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1,
                                value=st.session_state.inputs["pregnancies"])
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200,
                            value=st.session_state.inputs["glucose"])
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150,
                                    value=st.session_state.inputs["blood_pressure"])
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100,
                                    value=st.session_state.inputs["skin_thickness"])
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900,
                            value=st.session_state.inputs["insulin"])
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0,
                        value=st.session_state.inputs["bmi"])
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5,
                        value=st.session_state.inputs["dpf"])
    age = st.number_input("Age", min_value=1, max_value=120,
                        value=st.session_state.inputs["age"])
    
    if st.button("🔮 Predict Diabetes"):
        input_data = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "insulin": insulin,
            "bmi": bmi,
            "dpf": dpf,
            "age": age
        }
        
        result = call_api("/predict/diabetes", input_data)
        if result:
            prediction_label = result["prediction_label"]
            confidence = result["confidence"]
            
            if result["prediction"] == 1:
                st.error(f"✅ {prediction_label} (Confidence: {confidence:.2%})")
            else:
                st.success(f"❌ {prediction_label} (Confidence: {confidence:.2%})")

elif disease_option == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    
    # Get metrics from API
    metrics = get_metrics("heart")
    if metrics:
        st.markdown(
            f"**🧪 Accuracy:** {metrics['accuracy'] * 100:.2f}% | "
            f"**🎯 Precision:** {metrics['precision'] * 100:.2f}% | "
            f"**📡 Recall:** {metrics['recall'] * 100:.2f}% | "
            f"**📊 F1 Score:** {metrics['f1_score'] * 100:.2f}%"
        )
    
    # Preset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Fill Healthy Values"):
            st.session_state.inputs2 = healthy_heart.copy()
            st.rerun()
    with col2:
        if st.button("⚠️ Fill Heart-Risk Values"):
            st.session_state.inputs2 = unhealthy_heart.copy()
            st.rerun()
    
    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.inputs2["age"])
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1], index=[0, 1].index(st.session_state.inputs2["sex"]))
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=st.session_state.inputs2["cp"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=st.session_state.inputs2["trestbps"])
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=st.session_state.inputs2["chol"])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (0: No, 1: Yes)", [0, 1], index=[0, 1].index(st.session_state.inputs2["fbs"]))
    restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=st.session_state.inputs2["restecg"])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=st.session_state.inputs2["thalach"])
    exang = st.selectbox("Exercise Induced Angina (0: No, 1: Yes)", [0, 1], index=[0, 1].index(st.session_state.inputs2["exang"]))
    oldpeak = st.number_input("ST Depression (0.0 - 5.0)", min_value=0.0, max_value=5.0, value=st.session_state.inputs2["oldpeak"])
    slope = st.number_input("Slope of ST Segment (0-2)", min_value=0, max_value=2, value=st.session_state.inputs2["slope"])
    ca = st.number_input("Major Vessels Colored (0-4)", min_value=0, max_value=4, value=st.session_state.inputs2["ca"])
    thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=st.session_state.inputs2["thal"])
    
    if st.button("🔮 Predict Heart Disease"):
        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        
        result = call_api("/predict/heart", input_data)
        if result:
            prediction_label = result["prediction_label"]
            confidence = result["confidence"]
            
            if result["prediction"] == 1:
                st.error(f"✅ {prediction_label} (Confidence: {confidence:.2%})")
            else:
                st.success(f"❌ {prediction_label} (Confidence: {confidence:.2%})")

elif disease_option == "Parkinson's":
    st.subheader("Parkinson's Disease Prediction")
    
    # Get metrics from API
    metrics = get_metrics("parkinsons")
    if metrics:
        st.markdown(
            f"**🧪 Accuracy:** {metrics['accuracy'] * 100:.2f}% | "
            f"**🎯 Precision:** {metrics['precision'] * 100:.2f}% | "
            f"**📡 Recall:** {metrics['recall'] * 100:.2f}% | "
            f"**📊 F1 Score:** {metrics['f1_score'] * 100:.2f}%"
        )
    
    # Preset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Fill Healthy Values"):
            st.session_state.inputs3 = healthy_parkinsons.copy()
            st.rerun()
    with col2:
        if st.button("⚠️ Fill Parkinson-Risk Values"):
            st.session_state.inputs3 = unhealthy_parkinsons.copy()
            st.rerun()
    
    # Input fields (condensed for brevity)
    fo = st.number_input("MDVP:Fo(Hz)", min_value=50.0, max_value=300.0, value=st.session_state.inputs3["fo"])
    fhi = st.number_input("MDVP:Fhi(Hz)", min_value=50.0, max_value=600.0, value=st.session_state.inputs3["fhi"])
    flo = st.number_input("MDVP:Flo(Hz)", min_value=50.0, max_value=300.0, value=st.session_state.inputs3["flo"])
    jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.1, value=st.session_state.inputs3["jitter"])
    jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.1, value=st.session_state.inputs3["jitter_abs"])
    rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["rap"])
    ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["ppq"])
    ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["ddp"])
    shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.5, value=st.session_state.inputs3["shimmer"])
    shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=0.5, value=st.session_state.inputs3["shimmer_db"])
    apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["apq3"])
    apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["apq5"])
    apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["apq"])
    dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["dda"])
    nhr = st.number_input("NHR", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["nhr"])
    hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, value=st.session_state.inputs3["hnr"])
    rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["rpde"])
    dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["dfa"])
    spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0, value=st.session_state.inputs3["spread1"])
    spread2 = st.number_input("spread2", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["spread2"])
    d2 = st.number_input("D2", min_value=0.0, max_value=3.0, value=st.session_state.inputs3["d2"])
    ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, value=st.session_state.inputs3["ppe"])
    
    
    if st.button("🔮 Predict Parkinson's"):
        input_data = {
                "fo": fo,
                "fhi": fhi,
                "flo": flo,
                "jitter": jitter,
                "jitter_abs": jitter_abs,
                "rap": rap,
                "ppq": ppq,
                "ddp": ddp,
                "shimmer": shimmer,
                "shimmer_db": shimmer_db,
                "apq3": apq3,
                "apq5": apq5,
                "apq": apq,
                "dda": dda,
                "nhr": nhr,
                "hnr": hnr,
                "rpde": rpde,
                "dfa": dfa,
                "spread1": spread1,
                "spread2": spread2,
                "d2": d2,
                "ppe": ppe
            }
        
        result = call_api("/predict/parkinsons", input_data)
        if result:
            prediction_label = result["prediction_label"]
            confidence = result["confidence"]
            
            if result["prediction"] == 1:
                st.error(f"✅ {prediction_label} (Confidence: {confidence:.2%})")
            else:
                st.success(f"❌ {prediction_label} (Confidence: {confidence:.2%})")

        # Credit line 
        st.markdown("---")  # Separator for clarity
        st.markdown('<p style="text-align: center; font-size: 16px;">Developed by Mirang Bhandari 😊</p>', unsafe_allow_html=True)
