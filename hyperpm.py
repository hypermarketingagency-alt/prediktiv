import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="🧠 Neuromarketing ROAS Predictor", layout="wide")
st.title("🧠 Prediktív Neuromarketing Modell")

# ========== ADATOK BETÖLTÉSE ==========
st.sidebar.header("📊 Model Adatok")

# Alapértelmezett dummy data
use_custom = st.sidebar.checkbox("Saját adatok használata?")

if use_custom:
    uploaded_file = st.sidebar.file_uploader("CSV feltöltés", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ {len(df)} sor betöltve!")
    else:
        st.sidebar.warning("CSV szükséges!")
        st.stop()
else:
    # Dummy data (mint korábban)
    np.random.seed(42)
    n_samples = 1000
    data = {
        'platform_encoded': np.random.choice([0,1,2], n_samples),
        'emotion_score': np.random.uniform(0.1, 1.0, n_samples),
        # ... stb
    }
    df = pd.DataFrame(data)

# ========== MODEL TANÍTÁS ==========
@st.cache_resource
def train_model(data):
    features = ['platform_encoded', 'emotion_score', 'attention_score', 
                'social_proof', 'urgency_fomo', 'visual_contrast', 
                'personalization', 'budget', 'cpc', 'ctr']
    X = data[features]
    y = data['roas']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# ========== UI (mint korábban) ==========
col1, col2 = st.columns(2)
with col1:
    platform = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"])
    emotion = st.slider("Emotion Score", 0.0, 1.0, 0.7)
    attention = st.slider("Attention Score", 0.0, 1.0, 0.8)
# ... stb

