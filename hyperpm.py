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
            df['platform_encoded'] = 0
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

# ========== SZÖVEGELEMZÉS FUNKCIÓK ==========
def analyze_text(text):
    """Szövegelemzés - NLP alapú pontozás"""
    if not text:
        return 0.5, 0.5, 0, 0.5
    
    text_lower = text.lower()
    
    # Emotion Score - érzelmi szavak
    emotion_words = ['boldogság', 'szeretet', 'bizalom', 'biztonság', 'közösség', 'család', 
                     'mosolyog', 'szép', 'amazing', 'fantastic', 'love', 'happy', 'perfect',
                     'élj', 'végre', 'csoda', 'varázs', 'szív', 'kedves']
    emotion_count = sum(1 for word in emotion_words if word in text_lower)
    emotion_score = min(0.95, 0.3 + (emotion_count * 0.1))
    
    # Attention Score - figyelem szavak
    attention_words = ['azonnal', 'most', 'első', 'csak te', 'szenzációs', 'új', 'exkluzív',
                       'revolutionary', 'breakthrough', 'incredible', 'shocking', 'must-see',
                       'figyelj', 'vigyázz', 'különleges', 'ritka']
    attention_count = sum(1 for word in attention_words if word in text_lower)
    attention_score = min(0.95, 0.3 + (attention_count * 0.08))
    
    # Urgency/FOMO - sietség szavak
    urgency_words = ['most', 'azonnal', 'hamar', 'korlátozott', 'csak ma', 'utolsó', 'le fog járni',
                     'limited time', 'hurry', 'urgent', 'only', 'ends today', 'szabad hely vége',
                     'készlet limitált', 'ne maradj le', 'gyorsan', 'lezárás']
    urgency_fomo = 1 if any(word in text_lower for word in urgency_words) else 0
    
    # Personalization - személyesítési szavak
    personal_words = ['te', 'ön', 'neked', 'nekem', 'mi', 'te', 'személyes', 'custom',
                      'your', 'me', 'we', 'personal', 'unique']
    personal_count = sum(1 for word in personal_words if word in text_lower)
    personalization = min(0.95, 0.2 + (personal_count * 0.12))
    
    return emotion_score, attention_score, urgency_fomo, personalization

def analyze_image(image):
    """Képelemzés - egyszerű vizuális analízis"""
    try:
        img = Image.open(image).convert('RGB')
        
        # Képméret ellenőrzése
        width, height = img.size
        size_score = min(1.0, (width * height) / (1920 * 1080))
        
        # Szín analízis
        pixels = np.array(img.resize((100, 100)))
        r_mean, g_mean, b_mean = pixels[:,:,0].mean(), pixels[:,:,1].mean(), pixels[:,:,2].mean()
        
        # Kontraszt kalkuláció
        contrast = np.std(pixels) / 100
        visual_contrast = min(1.0, contrast)
        
        # Szín változatosság
        color_var = np.var([r_mean, g_mean, b_mean]) / 2000
        color_pop = min(1.0, color_var)
        
        # Attention score képből
        attention_from_image = (size_score * 0.5 + color_pop * 0.5)
        
        return visual_contrast, attention_from_image
    except Exception as e:
        st.warning(f"⚠️ Képelemzés hiba: {str(e)}")
        return 0.6, 0.6

# ========== TAB RENDSZER ==========
tab1, tab2 = st.tabs(["📊 Manuális Előrejelzés", "🖼️ Hirdetés Analyzer"])

# ==================== TAB 1: MANUÁLIS ELŐREJELZÉS ====================
with tab1:
    st.markdown("---")
    st.subheader("🎯 Hirdetés Paraméterei (Manuális)")
    
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
    budget = st.number_input("Hirdetési Költségvetés (USD)", 100, 50000, 2000, 100)
    cpc = st.number_input("Várható CPC (Cost Per Click) (USD)", 0.1, 10.0, 1.2, 0.1)
    ctr = st.number_input("Várható CTR (Click-Through Rate) (%)", 0.1, 15.0, 2.5, 0.1)
    
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
            'ctr': [ctr / 100]
        })
        
        roas_pred = model.predict(input_data)[0]
        revenue = budget * roas_pred
        profit = revenue - budget
        
        st.markdown("---")
        st.subheader("📊 Előrejelzés Eredménye")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Várható ROAS", f"{roas_pred:.2f}x", delta=f"+{roas_pred-1:.2f}x profit")
        with col2:
            st.metric("💵 Bevétel", f"${revenue:,.0f}", delta=f"+${profit:,.0f}")
        with col3:
            st.metric("🎯 CTR", f"{ctr:.1f}%")
        with col4:
            st.metric("💳 CPC", f"${cpc:.2f}")
        
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

# ==================== TAB 2: HIRDETÉS ANALYZER ====================
with tab2:
    st.markdown("---")
    st.subheader("🖼️ Hirdetés Automatikus Analízise")
    st.markdown("**Töltsd fel a hirdetésed képét és szövegét - az AI automatikusan pontozza!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Hirdetés Kép")
        uploaded_image = st.file_uploader("Válassz képet", type=["jpg", "jpeg", "png"])
        
        if uploaded_image:
            image_data = Image.open(uploaded_image)
            st.image(image_data, use_column_width=True)
            visual_contrast, attention_img = analyze_image(uploaded_image)
        else:
            visual_contrast, attention_img = 0.6, 0.6
    
    with col2:
        st.markdown("### 📝 Hirdetés Szöveg")
        ad_text = st.text_area("Másold ide a hirdetés szövegét", height=150, 
                               placeholder="Pl: 'Csoda módon új megoldás! Csak ma 50% kedvezmény!'")
        
        if ad_text:
            emotion_txt, attention_txt, urgency_txt, personal_txt = analyze_text(ad_text)
        else:
            emotion_txt, attention_txt, urgency_txt, personal_txt = 0.5, 0.5, 0, 0.5
    
    # ========== AUTO-PONTOZÁS ==========
    if uploaded_image or ad_text:
        st.markdown("---")
        st.subheader("🤖 Automatikus Pontozás (Jelenlegi Hirdetés)")
        
        emotion_score = min(0.95, (emotion_txt * 0.7 + attention_img * 0.3))
        attention_score = min(0.95, (attention_txt * 0.6 + visual_contrast * 0.4))
        urgency_fomo = urgency_txt
        personalization = personal_txt
        social_proof_auto = 5
        
        col1, col2 = st.columns(2)
        with col1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("❤️ Emotion Score", f"{emotion_score:.2f}/1.0")
            with col_b:
                st.metric("👁️ Attention Score", f"{attention_score:.2f}/1.0")
        
        with col2:
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("🎨 Visual Contrast", f"{visual_contrast:.2f}/1.0")
            with col_d:
                st.metric("🎯 Personalization", f"{personalization:.2f}/1.0")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👍 Social Proof", f"{social_proof_auto}/20")
        with col2:
            urgency_status = "✅ VAN" if urgency_fomo else "❌ NINCS"
            st.metric("⏰ FOMO/Urgency", urgency_status)
        
        st.markdown("---")
        st.subheader("💡 Elemzési Javaslatok")
        
        suggestions = []
        
        if emotion_score < 0.6:
            suggestions.append("📈 **Érzelmi elemek**: Adj erősebb érzelmi triggereket (szeretet, közösség)")
        
        if attention_score < 0.7:
            suggestions.append("👁️ **Figyelem**: Használj élénkebb szövegeket vagy nagyobb kontasztú képet")
        
        if personalization < 0.5:
            suggestions.append("🎯 **Personalizáció**: Adj hozzá személyesítési elemeket ('te', 'neked', 'egyedeid')")
        
        if urgency_fomo == 0:
            suggestions.append("⏰ **FOMO/Urgency**: Adj hozzá sietség-szavakat (most, hamar, korlátozott)")
        
        if visual_contrast < 0.6:
            suggestions.append("🎨 **Vizuális Kontraszt**: Használj élénkebb, magas kontrasztú képet")
        
        if suggestions:
            for sugg in suggestions:
                st.info(sugg)
        else:
            st.success("✅ Kiváló hirdetés! Jók az értékek!")
        
        # ========== ROAS ELŐREJELZÉS ==========
        st.markdown("---")
        col_calc1, col_calc2, col_calc3 = st.columns(3)
        
        with col_calc1:
            platform_auto = st.selectbox("Platform választása", ["Facebook", "Google Ads", "TikTok"], key="platform_auto")
        with col_calc2:
            budget_auto = st.number_input("Hirdetési Költségvetés (USD)", 100, 50000, 2000, 100, key="budget_auto")
        with col_calc3:
            cpc_auto = st.number_input("Várható CPC (USD)", 0.1, 10.0, 1.2, 0.1, key="cpc_auto")
        
        ctr_auto = 2.0 + (attention_score * 3)
        
        if st.button("🔮 ROAS Kalkulálás (Auto-Pontok)", type="primary", key="auto"):
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
            profit_current = revenue_current - budget_auto
            
            st.markdown("---")
            st.subheader("📊 Jelenlegi Hirdetés - Előrejelzés")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Várható ROAS", f"{roas_current:.2f}x", delta=f"+{roas_current-1:.2f}x profit")
            with col2:
                st.metric("💵 Bevétel", f"${revenue_current:,.0f}", delta=f"+${profit_current:,.0f}")
            with col3:
                st.metric("🎯 CTR", f"{ctr_auto:.1f}%")
            with col4:
                st.metric("💳 CPC", f"${cpc_auto:.2f}")
            
            # ========== WHAT-IF SIMULÁCIÓ ==========
            st.markdown("---")
            st.subheader("🚀 What-If Szimuláció - Javított Hirdetés")
            st.markdown("**Ha megvalósítod az alább javasolt módosításokat, itt az várható eredmény:**")
            
            emotion_improved = emotion_score
            attention_improved = attention_score
            urgency_improved = urgency_fomo
            personalization_improved = personalization
            visual_improved = visual_contrast
            
            if emotion_score < 0.7:
                emotion_improved = min(0.95, emotion_score + 0.15)
            if attention_score < 0.8:
                attention_improved = min(0.95, attention_score + 0.15)
            if urgency_fomo == 0:
                urgency_improved = 1
            if personalization < 0.6:
                personalization_improved = min(0.95, personalization + 0.15)
            if visual_contrast < 0.8:
                visual_improved = min(0.95, visual_contrast + 0.15)
            
            input_data_improved = pd.DataFrame({
                'platform_encoded': [plat_enc],
                'emotion_score': [emotion_improved],
                'attention_score': [attention_improved],
                'social_proof': [social_proof_auto],
                'urgency_fomo': [int(urgency_improved)],
                'visual_contrast': [visual_improved],
                'personalization': [personalization_improved],
                'budget': [budget_auto],
                'cpc': [cpc_auto],
                'ctr': [ctr_auto / 100]
            })
            
            roas_improved = model.predict(input_data_improved)[0]
            revenue_improved = budget_auto * roas_improved
            profit_improved = revenue_improved - budget_auto
            
            roas_delta = roas_improved - roas_current
            revenue_delta = revenue_improved - revenue_current
            profit_delta = profit_improved - profit_current
            roi_improvement = ((roas_improved - roas_current) / roas_current * 100) if roas_current > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Javított ROAS", f"{roas_improved:.2f}x", 
                         delta=f"+{roas_delta:.2f}x ({roi_improvement:+.1f}%)" if roas_delta != 0 else "Egyezés")
            with col2:
                st.metric("💵 Javított Bevétel", f"${revenue_improved:,.0f}", 
                         delta=f"+${revenue_delta:,.0f}" if revenue_delta > 0 else "Nincs változás")
            with col3:
                st.metric("📈 Extra Profit", f"${profit_delta:,.0f}", 
                         delta="🎯 Plusz nyereség" if profit_delta > 0 else "Egyezés")
            with col4:
                st.metric("✨ Javítás %", f"{roi_improvement:.1f}%" if roi_improvement > 0 else "—")
            
            st.markdown("---")
            st.subheader("📊 Részletes Összehasonlítás")
            
            comparison_df = pd.DataFrame({
                'Metrika': ['Emotion Score', 'Attention Score', 'Visual Contrast', 'Personalization', 'FOMO/Urgency'],
                'Jelenlegi': [f"{emotion_score:.2f}", f"{attention_score:.2f}", f"{visual_contrast:.2f}", 
                             f"{personalization:.2f}", "✅ VAN" if urgency_fomo else "❌ NINCS"],
                'Javított': [f"{emotion_improved:.2f}", f"{attention_improved:.2f}", f"{visual_improved:.2f}", 
                            f"{personalization_improved:.2f}", "✅ VAN"],
                'Javulás': [f"+{emotion_improved-emotion_score:.2f}", f"+{attention_improved-attention_score:.2f}", 
                           f"+{visual_improved-visual_contrast:.2f}", f"+{personalization_improved-personalization:.2f}", 
                           "✅ Hozzáadva" if urgency_improved > urgency_fomo else "—"]
            })
            
            st.table(comparison_df)

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
    
    ### Auto-Analyzer
    - **Szövegelemzés**: Érzelmi szavak, urgency trigger, personalizáció detectálása
    - **Képelemzés**: Szín kontraszt, méret, vizuális pop mérése
    - **What-If Szimuláció**: Megmutatja, mennyivel javulna a ROAS a javasolt módosítások után
    """)

with st.expander("📊 Minta CSV Format"):
    st.markdown("""
    ```
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
