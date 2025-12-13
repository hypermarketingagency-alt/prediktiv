import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🧠 Neuromarketing ROAS Predictor", layout="wide")

# ========== LOGO HOZZÁADÁSA ==========
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://raw.githubusercontent.com/hypermarketingagency-alt/prediktiv/main/logo.png", width=250)

st.title("🧠 Prediktív Neuromarketing Modell")
st.markdown("**FB/Google/TikTok ROAS optimalizálása**")

# ========== ADATOK BETÖLTÉSE ==========
st.sidebar.header("📊 Adatforrás Kiválasztása")

data_source = st.sidebar.radio(
    "Milyen adatokkal szeretnél tanítani?",
    ["Demo Adatok (Alapértelmezett)", "Saját CSV Feltöltés"]
)

@st.cache_resource
def load_demo_data():
    """Dummy adatok - alapértelmezett"""
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
        'budget': np.random.uniform(10000, 5000000, n_samples),
        'cpc': np.random.uniform(50, 3000, n_samples),
        'ctr': np.random.uniform(0.5, 5.0, n_samples)/100
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
    """Saját CSV adatok betöltése"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validálás - szükséges oszlopok
        required_cols = ['emotion_score', 'attention_score', 'social_proof', 'urgency_fomo',
                        'visual_contrast', 'personalization', 'budget', 'cpc', 'ctr', 'roas']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Hiányzó oszlopok: {', '.join(missing_cols)}")
            st.info(f"Szükséges oszlopok: {', '.join(required_cols)}")
            return None
        
        # Platform kódolás
        if 'platform' in df.columns:
            df['platform_encoded'] = df['platform'].map(
                {'Facebook': 0, 'Google Ads': 1, 'TikTok': 2}
            ).fillna(0).astype(int)
        else:
            df['platform_encoded'] = 0  # Default Facebook
            df['platform'] = 'Facebook'
        
        st.success(f"✅ {len(df)} sor sikeresen betöltve!")
        st.info(f"📊 Adatok: {df.shape[1]} oszlop, átlag ROAS: {df['roas'].mean():.2f}x")
        
        return df
    except Exception as e:
        st.error(f"❌ Hiba a CSV betöltésekor: {str(e)}")
        return None

# ========== ADATFORRÁS KIVÁLASZTÁSA ==========
if data_source == "Demo Adatok (Alapértelmezett)":
    st.sidebar.info("📌 Demo adatok használata - ideal teszteléshez")
    df = load_demo_data()
    data_mode = "demo"
else:
    st.sidebar.info("📁 Feltöltsd a saját CSV fájlodat")
    uploaded_file = st.sidebar.file_uploader(
        "CSV fájl feltöltése",
        type="csv",
        help="Szükséges oszlopok: emotion_score, attention_score, social_proof, urgency_fomo, visual_contrast, personalization, budget, cpc, ctr, roas"
    )
    
    if uploaded_file:
        df = load_custom_data(uploaded_file)
        if df is None:
            st.stop()
        data_mode = "custom"
    else:
        st.warning("⚠️ Kérjük, tölts fel egy CSV fájlt!")
        st.stop()

# ========== MODEL TANÍTÁS ==========
@st.cache_resource
def train_model(data):
    """Random Forest modell tanítása"""
    features = ['platform_encoded', 'emotion_score', 'attention_score', 'social_proof',
                'urgency_fomo', 'visual_contrast', 'personalization', 'budget', 'cpc', 'ctr']
    
    X = data[features].fillna(0)
    y = data['roas'].fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Model performance
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return model, rmse, r2, features

model, rmse, r2, features = train_model(df)

# ========== MODEL STATISZTIKA ==========
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Teljesítmény")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("R² Score", f"{r2:.3f}")
with col2:
    st.metric("RMSE", f"{rmse:.3f}")

if data_mode == "custom":
    st.sidebar.success("✅ Saját adatokkal tanítva!")
else:
    st.sidebar.info("ℹ️ Demo adatokkal tanítva")

# ========== ELŐREJELZÉS INPUTOK ==========
st.markdown("---")
st.subheader("🎯 Hirdetés Paraméterei")

col1, col2 = st.columns(2)
with col1:
    platform = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"])
    emotion = st.slider("Emotion Score (érzelmi engagement)", 0.0, 1.0, 0.7, 0.05)
    attention = st.slider("Attention Score (figyelemfelkeltő)", 0.0, 1.0, 0.8, 0.05)
    
with col2:
    social_proof = st.slider("Social Proof (testimonial/review)", 0, 20, 5)
    urgency = st.checkbox("FOMO/Urgency Element (pl. countdown, limited stock)")
    visual = st.slider("Visual Contrast (élénk színek)", 0.0, 1.0, 0.8, 0.05)

personal = st.slider("Personalizáció (név, dinamikus szöveg)", 0.0, 1.0, 0.6, 0.05)
budget = st.number_input("Hirdetési Költségvetés (HUF)", 10000, 5000000, 500000, 10000)
cpc = st.number_input("Várható CPC (Cost Per Click) (HUF)", 10, 1000, 300, 10)
ctr = st.number_input("Várható CTR (Click-Through Rate) (%)", 0.1, 15.0, 2.5, 0.1)

# ========== ELŐREJELZÉS ==========
if st.button("🔮 ROAS Előrejelzés & Optimalizálás", type="primary"):
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
        'ctr': [ctr / 100]  # Konvertálás %
    })
    
    roas_pred = model.predict(input_data)[0]
    revenue = budget * roas_pred
    profit = revenue - budget
    
    # ========== EREDMÉNYEK ==========
    st.markdown("---")
    st.subheader("📊 Előrejelzés Eredménye")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Várható ROAS", f"{roas_pred:.2f}x", delta=f"+{roas_pred-1:.2f}x profit")
    with col2:
        st.metric("💵 Bevétel", f"{revenue:,.0f} HUF", delta=f"+{profit:,.0f} HUF")
    with col3:
        st.metric("🎯 CTR", f"{ctr:.1f}%")
    with col4:
        st.metric("💳 CPC", f"{cpc:.0f} HUF")
    
    # ========== OPTIMALIZÁLÁSI JAVASLATOK ==========
    st.markdown("---")
    st.subheader("🚀 Neuromarketing Optimalizálások")
    
    recommendations = []
    
    if emotion < 0.7:
        recommendations.append({
            'icon': '📈',
            'title': 'Érzelmi Engagement Növelése',
            'desc': 'Erősítsd az érzelmi triggereket: boldogság, közösség, szeretet, biztonság',
            'impact': '+0.5-1.0x ROAS'
        })
    
    if attention < 0.8:
        recommendations.append({
            'icon': '👁️',
            'title': 'Figyelem Növelése Az Első 3 Másodpercben',
            'desc': 'Használj arcot (ez azonnal felismerhető), magas kontraszt, mozgás az elején',
            'impact': '+0.3-0.7x ROAS'
        })
    
    if social_proof < 5:
        recommendations.append({
            'icon': '👍',
            'title': 'Social Proof Maximalizálása',
            'desc': 'Adj hozzá testimonial videókat, 4.8⭐ értékeléseket, "500+ elégedett ügyfél" badget',
            'impact': '+0.4-0.6x ROAS'
        })
    
    if not urgency:
        recommendations.append({
            'icon': '⏰',
            'title': 'FOMO/Urgency Elem Hozzáadása',
            'desc': 'Countdown timer, "csak 3 db maradt", "48 óra akció", limited offer',
            'impact': '+0.3-0.5x ROAS'
        })
    
    if visual < 0.8:
        recommendations.append({
            'icon': '🎨',
            'title': 'Vizuális Pop Növelése',
            'desc': 'Élénk, kontrasztos színek, before-after képek, animációk',
            'impact': '+0.2-0.4x ROAS'
        })
    
    if personal < 0.6:
        recommendations.append({
            'icon': '🎯',
            'title': 'Personalizáció Javítása',
            'desc': 'Dinamikus szöveg (felhasználó neve), lokális referenciák, targeting finomítása',
            'impact': '+0.2-0.3x ROAS'
        })
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.write(rec['icon'])
            with col2:
                st.markdown(f"**{i}. {rec['title']}**")
                st.write(rec['desc'])
                st.caption(f"💡 Potenciális hatás: {rec['impact']}")
    else:
        st.success("✅ Kiváló paraméterek! Az ad már jól optimalizált!")
    
    # ========== BENCHMARK ==========
    st.markdown("---")
    st.subheader("📈 Benchmark Adatok")
    
    benchmark_data = {
        'Facebook': {'átlag_roas': 4.2, 'jó': 5.5, 'kiváló': 7.0},
        'Google Ads': {'átlag_roas': 3.8, 'jó': 5.0, 'kiváló': 6.5},
        'TikTok': {'átlag_roas': 5.2, 'jó': 6.8, 'kiváló': 8.5}
    }
    
    bench = benchmark_data[platform]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Átlag ROAS", f"{bench['átlag_roas']:.1f}x")
    with col2:
        st.metric("Jó ROAS", f"{bench['jó']:.1f}x")
    with col3:
        st.metric("Kiváló ROAS", f"{bench['kiváló']:.1f}x")
    with col4:
        if roas_pred >= bench['kiváló']:
            status = "🏆 KIVÁLÓ"
        elif roas_pred >= bench['jó']:
            status = "⭐ JÓ"
        elif roas_pred >= bench['átlag_roas']:
            status = "✓ ÁTLAG"
        else:
            status = "⚠️ FEJLESZTENDŐ"
        st.metric("Te", status)

# ========== HELP & INFO ==========
with st.expander("ℹ️ Hogyan működik a modell?"):
    st.markdown("""
    ### Random Forest Algoritmus
    Ez a modell **100 döntési fát** használ szavazási rendszerben:
    - Mindegyik fa más szöget lát az adatokra
    - Szavazatot ad a ROAS-ra
    - A végeredmény az összes fa átlaga
    
    ### Neuromarketing Tényezők
    - **Emotion Score**: Érzelmi engagement (0-1) - Az agy döntéseit érzelmek hajtják
    - **Attention Score**: Figyelem (0-1) - Az első 3 másodperc kritikus
    - **Social Proof**: Vélemények (0-20) - Emberek másolatnak
    - **FOMO/Urgency**: Sietség - Csökkenti a döntési időt
    - **Visual Contrast**: Szín (0-1) - Magas kontraszt = figyelem
    - **Personalization**: Egyéniesítés (0-1) - Név, lokálitás = magasabb CTR
    - **Budget**: Költségvetés - Nagyobb adspend = több impresszió
    - **CPC**: Kattintás ára - Platform határozza meg
    - **CTR**: Kattintási arány - Jó ad = 2-5% CTR
    
    ### Pontosság
    - **R² Score**: Mennyire pontosan jósol a modell (0-1)
    - **RMSE**: Átlagos hiba az előrejelzésben
    """)

with st.expander("📊 Minta CSV Format"):
    st.markdown("""
    ```
    platform,emotion_score,attention_score,social_proof,urgency_fomo,visual_contrast,personalization,budget,cpc,ctr,roas
    Facebook,0.75,0.82,8,1,0.85,0.7,500000,300,0.025,5.8
    Google Ads,0.65,0.78,5,0,0.75,0.6,400000,400,0.020,4.2
    TikTok,0.85,0.88,10,1,0.9,0.8,600000,200,0.035,7.1
    Facebook,0.7,0.75,8,1,0.8,0.65,550000,350,0.022,5.2
    ```
    
    **Szükséges oszlopok:**
    - emotion_score, attention_score, social_proof, urgency_fomo
    - visual_contrast, personalization, budget, cpc, ctr, roas
    
    **Opcionális:**
    - platform (Facebook/Google Ads/TikTok)
    """)

