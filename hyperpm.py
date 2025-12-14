import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

st.set_page_config(page_title="🧠 Neuromarketing ROAS Predictor", layout="wide")
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
def analyze_text(text):
    """Szövegelemzés - NLP alapú pontozás"""
    if not text:
        return 0.5, 0.5, 0, 0.5
    
    text_lower = text.lower()
    
    emotion_words = ['boldogság', 'szeretet', 'bizalom', 'biztonság', 'közösség', 'család', 
                     'mosolyog', 'szép', 'amazing', 'fantastic', 'love', 'happy', 'perfect']
    emotion_count = sum(1 for word in emotion_words if word in text_lower)
    emotion_score = min(0.95, 0.3 + (emotion_count * 0.1))
    
    attention_words = ['azonnal', 'most', 'első', 'szenzációs', 'új', 'exkluzív',
                       'revolutionary', 'breakthrough', 'incredible', 'shocking']
    attention_count = sum(1 for word in attention_words if word in text_lower)
    attention_score = min(0.95, 0.3 + (attention_count * 0.08))
    
    urgency_words = ['most', 'azonnal', 'hamar', 'korlátozott', 'csak ma', 'utolsó', 'le fog járni',
                     'limited time', 'hurry', 'urgent']
    urgency_fomo = 1 if any(word in text_lower for word in urgency_words) else 0
    
    personal_words = ['te', 'ön', 'neked', 'nekem', 'mi', 'személyes', 'custom', 'your', 'me', 'personal']
    personal_count = sum(1 for word in personal_words if word in text_lower)
    personalization = min(0.95, 0.2 + (personal_count * 0.12))
    
    return emotion_score, attention_score, urgency_fomo, personalization

def analyze_image(image):
    """Képelemzés - egyszerű vizuális analízis"""
    try:
        img = Image.open(image).convert('RGB')
        width, height = img.size
        size_score = min(1.0, (width * height) / (1920 * 1080))
        
        pixels = np.array(img.resize((100, 100)))
        r_mean, g_mean, b_mean = pixels[:,:,0].mean(), pixels[:,:,1].mean(), pixels[:,:,2].mean()
        
        contrast = np.std(pixels) / 100
        visual_contrast = min(1.0, contrast)
        
        color_var = np.var([r_mean, g_mean, b_mean]) / 2000
        color_pop = min(1.0, color_var)
        
        attention_from_image = (size_score * 0.5 + color_pop * 0.5)
        
        return visual_contrast, attention_from_image
    except Exception as e:
        st.warning(f"⚠️ Képelemzés hiba: {str(e)}")
        return 0.6, 0.6

st.markdown("""
<style>
.tooltip-container {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip-container .tooltip-icon {
    font-size: 16px;
    font-weight: bold;
    margin-left: 4px;
    padding: 2px 6px;
    border-radius: 50%;
    background-color: rgba(100, 200, 255, 0.2);
    transition: all 0.2s ease;
}

.tooltip-container .tooltip-icon:hover {
    background-color: rgba(100, 200, 255, 0.4);
    transform: scale(1.1);
}

.tooltip-container .tooltip-text {
    visibility: hidden;
    width: 280px;
    background-color: #1f2937;
    color: #fff;
    text-align: left;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 400;
    position: absolute;
    z-index: 1000;
    bottom: 120%;
    left: 50%;
    margin-left: -140px;
    opacity: 0;
    transition: opacity 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    line-height: 1.4;
}

.tooltip-container .tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #1f2937 transparent transparent transparent;
}

.tooltip-container:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

def tooltip_icon(text):
    """Hover tooltip generátor"""
    return f"""
    <span class="tooltip-container">
        <span class="tooltip-icon">ℹ️</span>
        <span class="tooltip-text">{text}</span>
    </span>
    """

tab1, tab2 = st.tabs(["📊 Manuális Előrejelzés", "🖼️ Hirdetés Analyzer"])

with tab1:
    st.markdown("---")
    st.subheader("🎯 Hirdetés Paraméterei (Manuális)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Platform** {tooltip_icon('Válaszd ki a platformot (Facebook, Google Ads vagy TikTok) - különböző algoritmusok és felhasználói viselkedés')}", unsafe_allow_html=True)
        platform = st.selectbox("Platform", ["Facebook", "Google Ads", "TikTok"], key="platform_manual", label_visibility="collapsed")
        
        st.markdown(f"**Emotion Score (Érzelmi Engagement)** {tooltip_icon('Mennyi érzelmi trigger van az adban (0=semleges, 1=nagyon érzelmes). Boldogság, szeretet, biztonság, közösség')}", unsafe_allow_html=True)
        emotion = st.slider("Emotion Score", 0.0, 1.0, 0.7, 0.05, key="emotion_manual", label_visibility="collapsed")
        
        st.markdown(f"**Attention Score (Figyelem)** {tooltip_icon('Mennyire vonz meg az ad a figyelmet (0=sárgaság, 1=szuperhatásos). Az első 3 másodperc dönt el mindent')}", unsafe_allow_html=True)
        attention = st.slider("Attention Score", 0.0, 1.0, 0.8, 0.05, key="attention_manual", label_visibility="collapsed")
        
    with col2:
        st.markdown(f"**Social Proof (Vélemények/Értékelések)** {tooltip_icon('Hány elégedett vásárlót említesz meg vagy mutatsz be az adban (0-20 értékelés/testimonial)')}", unsafe_allow_html=True)
        social_proof = st.slider("Social Proof", 0, 20, 5, key="social_proof_manual", label_visibility="collapsed")
        
        st.markdown(f"**FOMO/Urgency Element** {tooltip_icon('Van-e sietség érzés az adban? (Countdown, \"csak ma\", \"limitált készlet\", \"utolsó hely\")')}", unsafe_allow_html=True)
        urgency = st.checkbox("FOMO/Urgency Element", key="urgency_manual", label_visibility="collapsed")
        
        st.markdown(f"**Visual Contrast (Vizuális Kontraszt)** {tooltip_icon('Mennyire élénk és feltűnő a kép (0=unalmas, 1=nagyon kontraszt). Magas kontraszt = több kattintás')}", unsafe_allow_html=True)
        visual = st.slider("Visual Contrast", 0.0, 1.0, 0.8, 0.05, key="visual_manual", label_visibility="collapsed")
    
    st.markdown(f"**Personalizáció (Egyéniesítés)** {tooltip_icon('Hány személyesítési elem van az adban? (Felhasználó neve, \"neked\", \"te\", lokális referenciák)')}", unsafe_allow_html=True)
    personal = st.slider("Personalizáció", 0.0, 1.0, 0.6, 0.05, key="personal_manual", label_visibility="collapsed")
    
    st.markdown(f"**Hirdetési Költségvetés (HUF)** {tooltip_icon('Mennyit költesz az ad megjelenítésére (nagyobb budget = több impresszió és potenciális vásárló)')}", unsafe_allow_html=True)
    budget = st.number_input("Hirdetési Költségvetés (HUF)", 10000, 5000000, 500000, 10000, key="budget_manual", label_visibility="collapsed")
    
    st.markdown(f"**Várható CPC (Cost Per Click) (HUF)** {tooltip_icon('Átlagosan mennyibe kerül egy kattintás az adra (platform és verseny függvénye)')}", unsafe_allow_html=True)
    cpc = st.number_input("Várható CPC (HUF)", 10, 1000, 300, 10, key="cpc_manual", label_visibility="collapsed")
    
    st.markdown(f"**Várható CTR (Click-Through Rate) (%)** {tooltip_icon('Az összes lenyomásnak mekkora % fog rákattintani az adra (2-5% jó, 5%+ kiváló)')}", unsafe_allow_html=True)
    ctr = st.number_input("Várható CTR (%)", 0.1, 15.0, 2.5, 0.1, key="ctr_manual", label_visibility="collapsed")

# ========== ELŐREJELZÉS ==========
    if st.button("🔮 ROAS Előrejelzés & Optimalizálás", type="primary", key="manual"):
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
    with col1:        st.metric("💰 Várható ROAS", f"{roas_pred:.2f}x", delta=f"+{roas_pred-1:.2f}x profit")
        with col2:
            st.metric("💵 Bevétel", f"{revenue:,.0f} HUF", delta=f"+{profit:,.0f} HUF")
    with col3:
        st.metric("🎯 CTR", f"{ctr:.1f}%")        with col4:
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
            'desc': 'Élénk, kontrasztos színek, before-after képek,animációk',
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
    ```csv
    platform,emotion_score,attention_score,social_proof,urgency_fomo,visual_contrast,personalization,budget,cpc,ctr,roas
    Facebook,0.75,0.82,8,1,0.85,0.7,2000,1.2,0.025,5.8
    Google Ads,0.65,0.78,5,0,0.75,0.6,1500,1.5,0.020,4.2
    TikTok,0.85,0.88,10,1,0.9,0.8,3000,0.8,0.035,7.1
    Facebook,0.7,0.75,8,1,0.8,0.65,2500,1.1,0.022,5.2
    ```
    
    **Szükséges oszlopok:**
    - emotion_score, attention_score, social_proof, urgency_fomo
    - visual_contrast, personalization, budget, cpc, ctr, roas
    
    **Opcionális:**
    - platform (Facebook/Google Ads/TikTok)
    """)
