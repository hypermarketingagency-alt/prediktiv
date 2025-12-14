import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

st.set_page_config(page_title="🧠 Neuromarketing ROAS Predictor", layout="wide")
st.title("🧠 Prediktív Neuromarketing Modell")
st.markdown("**FB/Google/TikTok ROAS optimalizálása**")

st.sidebar.header("📊 Adatforrás Kiválasztása")

data_source = st.sidebar.radio(
    "Milyen adatokkal szeretnél tanítani?",
    ["Demo Adatok (Alapértelmezett)", "Saját CSV Feltöltés"]
)

@st.cache_resource
def load_demo_data():
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
    df['platform'] = df['platform_encoded'].map({0: 'Facebook', 1: 'Google Ads', 2: 'TikTok'})
    return df

def load_custom_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['emotion_score', 'attention_score', 'social_proof', 'urgency_fomo',
                        'visual_contrast', 'personalization', 'budget', 'cpc', 'ctr', 'roas']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Hiányzó oszlopok: {', '.join(missing_cols)}")
            return None
        
        if 'platform' in df.columns:
            df['platform_encoded'] = df['platform'].map(
                {'Facebook': 0, 'Google Ads': 1, 'TikTok': 2}
            ).fillna(0).astype(int)
        else:
            df['platform_encoded'] = 0
            df['platform'] = 'Facebook'
        
        st.success(f"✅ {len(df)} sor sikeresen betöltve!")
        return df
    except Exception as e:
        st.error(f"❌ Hiba: {str(e)}")
        return None

if data_source == "Demo Adatok (Alapértelmezett)":
    st.sidebar.info("📌 Demo adatok")
    df = load_demo_data()
    data_mode = "demo"
else:
    st.sidebar.info("📁 CSV fájl feltöltése")
    uploaded_file = st.sidebar.file_uploader("CSV fájl feltöltése", type="csv")
    
    if uploaded_file:
        df = load_custom_data(uploaded_file)
        if df is None:
            st.stop()
        data_mode = "custom"
    else:
        st.warning("⚠️ Kérjük, tölts fel egy CSV fájlt!")
        st.stop()

@st.cache_resource
def train_model(data):
    features = ['platform_encoded', 'emotion_score', 'attention_score', 'social_proof',
                'urgency_fomo', 'visual_contrast', 'personalization', 'budget', 'cpc', 'ctr']
    
    X = data[features].fillna(0)
    y = data['roas'].fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return model, rmse, r2, features

model, rmse, r2, features = train_model(df)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Teljesítmény")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("R² Score", f"{r2:.3f}")
with col2:
    st.metric("RMSE", f"{rmse:.3f}")

def analyze_text(text):
    if not text:
        return 0.5, 0.5, 0, 0.5
    
    text_lower = text.lower()
    emotion_words = ['boldogság', 'szeretet', 'bizalom', 'biztonság', 'közösség', 'család', 'szép', 'amazing', 'love', 'happy']
    emotion_count = sum(1 for word in emotion_words if word in text_lower)
    emotion_score = min(0.95, 0.3 + (emotion_count * 0.1))
    
    attention_words = ['azonnal', 'most', 'szenzációs', 'új', 'exkluzív', 'revolutionary']
    attention_count = sum(1 for word in attention_words if word in text_lower)
    attention_score = min(0.95, 0.3 + (attention_count * 0.08))
    
    urgency_words = ['most', 'azonnal', 'hamar', 'korlátozott', 'csak ma', 'limited time']
    urgency_fomo = 1 if any(word in text_lower for word in urgency_words) else 0
    
    personal_words = ['te', 'neked', 'nekem', 'mi', 'your', 'personal']
    personal_count = sum(1 for word in personal_words if word in text_lower)
    personalization = min(0.95, 0.2 + (personal_count * 0.12))
    
    return emotion_score, attention_score, urgency_fomo, personalization

def analyze_image(image):
    try:
        img = Image.open(image).convert('RGB')
        width, height = img.size
        size_score = min(1.0, (width * height) / (1920 * 1080))
        
        pixels = np.array(img.resize((100, 100)))
        contrast = np.std(pixels) / 100
        visual_contrast = min(1.0, contrast)
        
        r_mean, g_mean, b_mean = pixels[:,:,0].mean(), pixels[:,:,1].mean(), pixels[:,:,2].mean()
        color_var = np.var([r_mean, g_mean, b_mean]) / 2000
        color_pop = min(1.0, color_var)
        
        attention_from_image = (size_score * 0.5 + color_pop * 0.5)
        return visual_contrast, attention_from_image
    except:
        return 0.6, 0.6

st.markdown("""<style>
.tooltip-container { position: relative; display: inline-block; cursor: help; }
.tooltip-container .tooltip-icon { font-size: 14px; margin-left: 4px; }
.tooltip-container .tooltip-text { visibility: hidden; width: 280px; background-color: #1f2937; color: #fff; text-align: left; padding: 10px 12px; border-radius: 6px; font-size: 11px; position: absolute; z-index: 1000; bottom: 120%; left: 50%; margin-left: -140px; opacity: 0; transition: opacity 0.3s; border: 1px solid rgba(255,255,255,0.2); line-height: 1.4; }
.tooltip-container .tooltip-text::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #1f2937 transparent transparent transparent; }
.tooltip-container:hover .tooltip-text { visibility: visible; opacity: 1; }
</style>""", unsafe_allow_html=True)

def tooltip_icon(text):
    return f'<span class="tooltip-container"><span class="tooltip-icon">ℹ️</span><span class="tooltip-text">{text}</span></span>'

tab1, tab2 = st.tabs(["📊 Manuális Előrejelzés", "🖼️ Hirdetés Analyzer"])

with tab1:
    st.markdown("---")
    st.subheader("🎯 Hirdetés Paraméterei")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Platform** {tooltip_icon('Válaszd ki a platformot')}", unsafe_allow_html=True)
        platform = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"], key="platform_manual", label_visibility="collapsed")
        
        st.markdown(f"**Emotion Score** {tooltip_icon('Érzelmi trigger mennyisége')}", unsafe_allow_html=True)
        emotion = st.slider("Emotion Score", 0.0, 1.0, 0.7, 0.05, key="emotion_manual", label_visibility="collapsed")
        
        st.markdown(f"**Attention Score** {tooltip_icon('Figyelem vonzása')}", unsafe_allow_html=True)
        attention = st.slider("Attention Score", 0.0, 1.0, 0.8, 0.05, key="attention_manual", label_visibility="collapsed")
        
    with col2:
        st.markdown(f"**Social Proof** {tooltip_icon('Vélemények száma')}", unsafe_allow_html=True)
        social_proof = st.slider("Social Proof", 0, 20, 5, key="social_proof_manual", label_visibility="collapsed")
        
        st.markdown(f"**FOMO/Urgency** {tooltip_icon('Sietség érzés')}", unsafe_allow_html=True)
        urgency = st.checkbox("FOMO/Urgency Element", key="urgency_manual", label_visibility="collapsed")
        
        st.markdown(f"**Visual Contrast** {tooltip_icon('Képek élénksége')}", unsafe_allow_html=True)
        visual = st.slider("Visual Contrast", 0.0, 1.0, 0.8, 0.05, key="visual_manual", label_visibility="collapsed")
    
    st.markdown(f"**Personalizáció** {tooltip_icon('Személyesítési elemek')}", unsafe_allow_html=True)
    personal = st.slider("Personalizáció", 0.0, 1.0, 0.6, 0.05, key="personal_manual", label_visibility="collapsed")
    
    st.markdown(f"**Budget (HUF)** {tooltip_icon('Hirdetési költségvetés')}", unsafe_allow_html=True)
    budget = st.number_input("Hirdetési Költségvetés", 10000, 5000000, 500000, 10000, key="budget_manual", label_visibility="collapsed")
    
    st.markdown(f"**CPC (HUF)** {tooltip_icon('Egy kattintás ára')}", unsafe_allow_html=True)
    cpc = st.number_input("Várható CPC", 10, 1000, 300, 10, key="cpc_manual", label_visibility="collapsed")
    
    st.markdown(f"**CTR (%)** {tooltip_icon('Kattintási arány')}", unsafe_allow_html=True)
    ctr = st.number_input("Várható CTR", 0.1, 15.0, 2.5, 0.1, key="ctr_manual", label_visibility="collapsed")

    if st.button("🔮 ROAS Előrejelzés", type="primary", key="manual"):
        plat_enc = {"Facebook": 0, "Google Ads": 1, "TikTok": 2}[platform]
        
        input_data = pd.DataFrame({
            'platform_encoded': [plat_enc],
            'emotion_score': [emotion],
            'attention_score': [attention],
            'social_proof': [social_proof],
            'urgency_fomo': [int(urgency)],
            'visual_contrast': [visual],
            'personalization': [personal],
            'budget': [budget],
            'cpc': [cpc],
            'ctr': [ctr / 100]
        })
        
        roas_pred = model.predict(input_data)[0]
        revenue = budget * roas_pred
        profit = revenue - budget
        
        st.markdown("---")
        st.subheader("📊 Előrejelzés Eredménye")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 ROAS", f"{roas_pred:.2f}x")
        with col2:
            st.metric("💵 Bevétel", f"{revenue:,.0f} HUF")
        with col3:
            st.metric("🎯 CTR", f"{ctr:.1f}%")
        with col4:
            st.metric("💳 CPC", f"{cpc:.0f} HUF")

with tab2:
    st.markdown("---")
    st.subheader("🖼️ Hirdetés Automatikus Analízise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**📸 Hirdetés Kép** {tooltip_icon('JPG/PNG formátum')}", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Képfeltöltés", type=["jpg", "jpeg", "png"], key="image_analyzer", label_visibility="collapsed")
        
        if uploaded_image:
            image_data = Image.open(uploaded_image)
            st.image(image_data, use_column_width=True)
            visual_contrast, attention_img = analyze_image(uploaded_image)
        else:
            visual_contrast, attention_img = 0.6, 0.6
    
    with col2:
        st.markdown(f"**📝 Hirdetés Szöveg** {tooltip_icon('Másold ide a szöveget')}", unsafe_allow_html=True)
        ad_text = st.text_area("Szöveg", height=150, placeholder="Pl: Csoda módon új megoldás! Csak ma 50% kedvezmény!", key="text_analyzer", label_visibility="collapsed")
        
        if ad_text:
            emotion_txt, attention_txt, urgency_txt, personal_txt = analyze_text(ad_text)
        else:
            emotion_txt, attention_txt, urgency_txt, personal_txt = 0.5, 0.5, 0, 0.5
    
    if uploaded_image or ad_text:
        st.markdown("---")
        st.subheader("🤖 Automatikus Pontozás")
        
        emotion_score = min(0.95, (emotion_txt * 0.7 + attention_img * 0.3))
        attention_score = min(0.95, (attention_txt * 0.6 + visual_contrast * 0.4))
        urgency_fomo = urgency_txt
        personalization = personal_txt
        social_proof_auto = 5
        
        col1, col2 = st.columns(2)
        with col1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("❤️ Emotion", f"{emotion_score:.2f}")
            with col_b:
                st.metric("👁️ Attention", f"{attention_score:.2f}")
        
        with col2:
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("🎨 Visual", f"{visual_contrast:.2f}")
            with col_d:
                st.metric("🎯 Personal", f"{personalization:.2f}")
        
        st.markdown("---")
        col_calc1, col_calc2, col_calc3 = st.columns(3)
        
        with col_calc1:
            st.markdown(f"**Platform** {tooltip_icon('Melyik platformon?')}", unsafe_allow_html=True)
            platform_auto = st.selectbox("Platform2", ["Facebook", "Google Ads", "TikTok"], key="platform_analyzer", label_visibility="collapsed")
        
        with col_calc2:
            st.markdown(f"**Budget** {tooltip_icon('Költségvetés')}", unsafe_allow_html=True)
            budget_auto = st.number_input("Budget2", 10000, 5000000, 500000, 10000, key="budget_analyzer", label_visibility="collapsed")
        
        with col_calc3:
            st.markdown(f"**CPC** {tooltip_icon('Kattintás ára')}", unsafe_allow_html=True)
            cpc_auto = st.number_input("CPC2", 10, 1000, 300, 10, key="cpc_analyzer", label_visibility="collapsed")
        
        ctr_auto = 2.0 + (attention_score * 3)
        
        if st.button("🔮 ROAS Kalkulálás", type="primary", key="auto"):
            plat_enc = {"Facebook": 0, "Google Ads": 1, "TikTok": 2}[platform_auto]
            
            input_data = pd.DataFrame({
                'platform_encoded': [plat_enc],
                'emotion_score': [emotion_score],
                'attention_score': [attention_score],
                'social_proof': [social_proof_auto],
                'urgency_fomo': [int(urgency_fomo)],
                'visual_contrast': [visual_contrast],
                'personalization': [personalization],
                'budget': [budget_auto],
                'cpc': [cpc_auto],
                'ctr': [ctr_auto / 100]
            })
            
            roas_current = model.predict(input_data)[0]
            revenue_current = budget_auto * roas_current
            
            st.markdown("---")
            st.subheader("📊 Jelenlegi Hirdetés")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 ROAS", f"{roas_current:.2f}x")
            with col2:
                st.metric("💵 Bevétel", f"{revenue_current:,.0f} HUF")
            with col3:
                st.metric("🎯 CTR", f"{ctr_auto:.1f}%")
            with col4:
                st.metric("💳 CPC", f"{cpc_auto:.0f} HUF")

with st.expander("ℹ️ Hogyan működik?"):
    st.markdown("""
    ### Random Forest Algoritmus
    - 100 döntési fa
    - Szavazási rendszer
    
    ### Tényezők
    - Érzelmi engagement
    - Figyelem
    - Vélemények
    - FOMO/Urgency
    - Vizuális kontraszt
    - Személyesítés
    """)
