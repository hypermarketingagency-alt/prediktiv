import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="🧠 Neuromarketing ROAS Predictor", layout="wide")
st.title("🧠 Prediktív Neuromarketing Modell")
st.markdown("**FB/Google/TikTok ROAS optimalizálása**")

@st.cache_resource
def load_model():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'platform_encoded': np.random.choice([0,1,2], n_samples),
        'emotion_score': np.random.uniform(0.1, 1.0, n_samples),
        'attention_score': np.random.uniform(0.2, 0.95, n_samples),
        'social_proof': np.random.choice([3,5,10,20], n_samples, p=[0.3,0.4,0.2,0.1]),
        'urgency_fomo': np.random.choice([0,1], n_samples, p=[0.6,0.4]),
        'visual_contrast': np.random.uniform(0.5, 1.0, n_samples),
        'personalization': np.random.uniform(0,1,n_samples),
        'budget': np.random.uniform(100,5000,n_samples),
        'cpc': np.random.uniform(0.5,3.0,n_samples),
        'ctr': np.random.uniform(0.5,5.0,n_samples)/100
    }
    neuromarketing_factor = (data['emotion_score']*0.3 + data['attention_score']*0.25 + 
                            np.log(data['social_proof']+1)*0.15 + data['urgency_fomo']*0.1 + 
                            data['visual_contrast']*0.1 + data['personalization']*0.1)
    data['roas'] = np.clip(2 + neuromarketing_factor*4 + np.log(data['budget'])*0.1 + 
                          data['ctr']*20 + np.random.normal(0,0.5,n_samples), 1.0, 10.0)
    df = pd.DataFrame(data)
    X = df.drop('roas', axis=1)
    y = df['roas']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

col1, col2 = st.columns(2)
with col1:
    platform = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"])
    emotion = st.slider("Emotion Score", 0.0, 1.0, 0.7)
    attention = st.slider("Attention Score", 0.0, 1.0, 0.8)
with col2:
    social_proof = st.slider("Social Proof", 0, 20, 5)
    urgency = st.checkbox("FOMO/Urgency")
    visual = st.slider("Visual Contrast", 0.0, 1.0, 0.8)

personal = st.slider("Personalizáció", 0.0, 1.0, 0.6)
budget = st.number_input("Budget (USD)", 100, 10000, 1000)
cpc = st.number_input("CPC", 0.1, 5.0, 1.0)
ctr = st.number_input("CTR (%)", 0.1, 10.0, 2.0) / 100

if st.button("🔮 Előrejelzés", type="primary"):
    plat_enc = {"Facebook":0, "Google Ads":1, "TikTok":2}[platform]
    input_data = pd.DataFrame({
        'platform_encoded':[plat_enc],'emotion_score':[emotion],'attention_score':[attention],
        'social_proof':[social_proof],'urgency_fomo':[int(urgency)],'visual_contrast':[visual],
        'personalization':[personal],'budget':[budget],'cpc':[cpc],'ctr':[ctr]
    })
    roas_pred = model.predict(input_data)[0]
    st.success(f"**Várható ROAS: {roas_pred:.2f}x**")
    
    st.info("**Optimalizációk:**")
    if emotion < 0.7: st.success("📈 Erősítsd érzelmi triggereket")
    if attention < 0.8: st.success("👁️ Arcok első 3s-ben")
    if social_proof < 5: st.success("👍 Több testimonial")
    if not urgency: st.success("⏰ FOMO elem")
