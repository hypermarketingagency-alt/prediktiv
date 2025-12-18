import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from thefuzz import fuzz
except ImportError:
    st.error("Hiányzik: pip install thefuzz python-Levenshtein")
    st.stop()

import json
import io

# ============================================================================
# 🎨 HYPER App - Neuromarketing ROAS Predictor v3.0
# FÁZIS 1: CSV Importer & Intelligent Mapper
# ============================================================================

st.set_page_config(
    page_title="HYPER - Marketing Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 📊 CONFIGURATION & MAPPINGS
# ============================================================================

UNIFIED_SCHEMA = {
    "mandatory": [
        ("date_start", "date", "Jelentés kezdete (dátum)"),
        ("date_end", "date", "Jelentés vége (dátum)"),
        ("campaign_name", "string", "Kampány neve"),
        ("platform", "string", "Platform (Facebook/Google Ads/TikTok)"),
        ("campaign_status", "string", "Kampány státusza"),
        ("spend", "float", "Elköltött összeg (HUF)"),
        ("conversions", "int", "Konverziók / Vásárlások"),
        ("conversion_value", "float", "Konverziós érték (HUF)"),
    ],
    "recommended": [
        ("impressions", "int", "Megjelenések"),
        ("clicks", "int", "Kattintások / Interakciók"),
        ("ctr_percent", "float", "CTR (%)"),
        ("cpc", "float", "CPC (HUF)"),
        ("cpa", "float", "CPA (HUF)"),
        ("roas", "float", "ROAS"),
        ("reach", "int", "Elérés"),
        ("frequency", "float", "Gyakoriság"),
        ("ad_group_name", "string", "Ad Set / Ad Group neve"),
        ("budget_type", "string", "Költségkeret típusa"),
        ("budget_allocated", "float", "Költségkeret (HUF)"),
    ],
    "optional": [
        ("add_to_cart", "int", "Kosárba helyezések"),
        ("video_views", "int", "Videó megtekintések"),
        ("engagement", "int", "Engagement"),
        ("conversion_type", "string", "Konverzió típusa"),
        ("notes", "string", "Megjegyzések"),
    ]
}

# Fuzzy matching patterns
COLUMN_PATTERNS = {
    # Spend related
    "spend": ["elköltött", "költség", "spend", "ad spend", "expense", "amount spent"],
    
    # Conversions
    "conversions": ["vásárlás", "konverzi", "conversion", "purchase", "order", "sale"],
    
    # Conversion value
    "conversion_value": ["konverziós érték", "érték", "revenue", "value", "bevétel", "sales value"],
    
    # Impressions
    "impressions": ["megjelenés", "impression", "views", "display"],
    
    # Clicks
    "clicks": ["kattintás", "click", "link click", "interakci"],
    
    # CTR
    "ctr_percent": ["ctr", "átkattintási"],
    
    # CPC
    "cpc": ["cpc", "cost per click", "költség/kattintás"],
    
    # CPA
    "cpa": ["cpa", "költség/konv", "cost per acquisition", "cost per conversion", "eredményen", "acquisition cost"],
    
    # ROAS
    "roas": ["roas", "hirdetésmegtérülés", "return on ad spend", "megtérülés"],
    
    # Reach
    "reach": ["elérés", "reach", "unique reach"],
    
    # Frequency
    "frequency": ["gyakoriság", "frequency", "avg frequency"],
    
    # Campaign name
    "campaign_name": ["kampány", "campaign"],
    
    # Campaign status
    "campaign_status": ["státusz", "status", "state", "enabled", "active"],
    
    # Platform
    "platform": ["platform", "csatorna", "channel", "channel type"],
    
    # Date
    "date_start": ["kezdete", "start", "from", "report start"],
    "date_end": ["vége", "end", "to", "report end"],
    
    # Add to cart
    "add_to_cart": ["kosárba", "add to cart", "cart addition"],
    
    # Video views
    "video_views": ["videó", "video view", "video play", "video watch"],
}

# ============================================================================
# 🔧 HELPER FUNCTIONS
# ============================================================================

def find_matching_column(csv_column, patterns_dict, threshold=70):
    """Fuzzy match CSV column to unified schema"""
    csv_col_lower = csv_column.lower().strip()
    
    best_match = None
    best_score = 0
    
    for unified_field, patterns in patterns_dict.items():
        for pattern in patterns:
            score = fuzz.partial_ratio(csv_col_lower, pattern.lower())
            if score > best_score:
                best_score = score
                best_match = unified_field
    
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


def intelligently_map_columns(df_columns):
    """Create mapping from CSV columns to unified schema"""
    mapping = {}
    unmapped = []
    
    for col in df_columns:
        matched_field, score = find_matching_column(col, COLUMN_PATTERNS)
        if matched_field:
            mapping[col] = matched_field
        else:
            unmapped.append(col)
    
    return mapping, unmapped


def parse_numeric_value(val):
    """Parse Hungarian-formatted numbers"""
    if pd.isna(val) or val == '' or val == '–' or val == '--':
        return np.nan
    
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip()
    val_str = val_str.replace(" ", "").replace(",", ".")
    
    try:
        return float(val_str)
    except:
        return np.nan


def parse_percentage(val):
    """Parse percentage values"""
    if pd.isna(val) or val == '' or val == '–':
        return np.nan
    
    val_str = str(val).strip()
    val_str = val_str.replace("%", "").replace(",", ".")
    
    try:
        pct = float(val_str)
        # If value is > 1, assume it's already a percentage (not decimal)
        return pct if pct <= 100 else pct / 100
    except:
        return np.nan


def parse_date(val):
    """Parse date values"""
    if pd.isna(val):
        return None
    
    date_formats = [
        "%Y-%m-%d",
        "%Y.%m.%d",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            continue
    
    try:
        return pd.to_datetime(val)
    except:
        return None


def normalize_data(df, mapping, user_adjustments=None):
    """Normalize and clean imported data"""
    
    # Apply user adjustments if provided
    if user_adjustments:
        mapping = {**mapping, **user_adjustments}
    
    # Create unified dataframe
    normalized_df = pd.DataFrame()
    
    for csv_col, unified_col in mapping.items():
        if csv_col not in df.columns:
            continue
        
        # Get the field info
        field_info = None
        for section in [UNIFIED_SCHEMA["mandatory"], UNIFIED_SCHEMA["recommended"], UNIFIED_SCHEMA["optional"]]:
            for field in section:
                if field[0] == unified_col:
                    field_info = field
                    break
        
        if not field_info:
            continue
        
        field_name, field_type, _ = field_info
        raw_data = df[csv_col]
        
        # Apply type-specific parsing
        if field_type == "float":
            normalized_df[field_name] = raw_data.apply(parse_numeric_value)
        elif field_type == "int":
            normalized_df[field_name] = raw_data.apply(lambda x: int(parse_numeric_value(x)) if not pd.isna(parse_numeric_value(x)) else np.nan)
        elif field_type == "date":
            normalized_df[field_name] = raw_data.apply(parse_date)
        elif field_type == "string":
            normalized_df[field_name] = raw_data.astype(str)
        else:
            normalized_df[field_name] = raw_data
    
    # Calculate missing metrics
    if "spend" in normalized_df.columns and "conversion_value" in normalized_df.columns:
        if "roas" not in normalized_df.columns:
            normalized_df["roas"] = normalized_df["conversion_value"] / normalized_df["spend"]
            normalized_df["roas"] = normalized_df["roas"].replace([np.inf, -np.inf], np.nan)
    
    if "spend" in normalized_df.columns and "conversions" in normalized_df.columns:
        if "cpa" not in normalized_df.columns:
            normalized_df["cpa"] = normalized_df["spend"] / normalized_df["conversions"]
            normalized_df["cpa"] = normalized_df["cpa"].replace([np.inf, -np.inf], np.nan)
    
    if "clicks" in normalized_df.columns and "impressions" in normalized_df.columns:
        if "ctr_percent" not in normalized_df.columns:
            normalized_df["ctr_percent"] = (normalized_df["clicks"] / normalized_df["impressions"] * 100)
            normalized_df["ctr_percent"] = normalized_df["ctr_percent"].replace([np.inf, -np.inf], np.nan)
    
    if "spend" in normalized_df.columns and "clicks" in normalized_df.columns:
        if "cpc" not in normalized_df.columns:
            normalized_df["cpc"] = normalized_df["spend"] / normalized_df["clicks"]
            normalized_df["cpc"] = normalized_df["cpc"].replace([np.inf, -np.inf], np.nan)
    
    return normalized_df


def validate_data(df):
    """Validate normalized data"""
    issues = []
    
    # Check mandatory fields
    mandatory_fields = [f[0] for f in UNIFIED_SCHEMA["mandatory"]]
    for field in mandatory_fields:
        if field not in df.columns:
            issues.append(f"❌ Hiányzik: {field}")
        elif df[field].isna().sum() > len(df) * 0.5:
            issues.append(f"⚠️ Túl sok hiányzik: {field} ({df[field].isna().sum()} / {len(df)})")
    
    # Check value ranges
    if "roas" in df.columns:
        invalid_roas = df[(df["roas"] < 0) | (df["roas"] > 100)].shape[0]
        if invalid_roas > 0:
            issues.append(f"⚠️ Érvénytelen ROAS értékek: {invalid_roas}")
    
    if "cpa" in df.columns:
        invalid_cpa = df[(df["cpa"] < 0)].shape[0]
        if invalid_cpa > 0:
            issues.append(f"⚠️ Negatív CPA értékek: {invalid_cpa}")
    
    return issues


# ============================================================================
# 🎨 STREAMLIT UI
# ============================================================================

st.title("🎯 HYPER - Marketing Campaign Analyzer")
st.markdown("### Fázis 1: Intelligens CSV Importer")

# Initialize session state
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "mapping" not in st.session_state:
    st.session_state.mapping = {}
if "normalized_data" not in st.session_state:
    st.session_state.normalized_data = None

# ============================================================================
# TAB 1: UPLOAD & MAPPING
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📥 Feltöltés & Mapping", "✅ Validáció", "📊 Előnézet", "💾 Mentés"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1️⃣ CSV/Excel Feltöltés")
        uploaded_file = st.file_uploader(
            "Válassz CSV vagy Excel fájlt",
            type=["csv", "xlsx", "xls"],
            help="Facebook, Google Ads vagy TikTok export"
        )
    
    with col2:
        st.subheader("ℹ️ Támogatott formátumok")
        st.markdown("""
        - ✅ Facebook Ads Manager
        - ✅ Google Ads
        - ⏳ TikTok (hamarosan)
        """)
    
    if uploaded_file:
        try:
            # Load file
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = raw_df
            
            st.success(f"✅ Betöltve: {uploaded_file.name}")
            st.info(f"📊 Sorok: {len(raw_df)}, Oszlopok: {len(raw_df.columns)}")
            
            # ====================================================================
            # INTELLIGENT COLUMN MAPPING
            # ====================================================================
            
            st.subheader("2️⃣ Automata Oszlop Felismerés")
            
            initial_mapping, unmapped = intelligently_map_columns(raw_df.columns)
            
            st.session_state.mapping = initial_mapping
            
            st.markdown("#### 🔄 Automatikusan felismert oszlopok:")
            
            # Show mapped columns
            mapped_cols = st.expander("✅ Leképezett oszlopok", expanded=True)
            with mapped_cols:
                mapping_display = []
                for csv_col, unified_col in sorted(initial_mapping.items()):
                    mapping_display.append({
                        "CSV Oszlop": csv_col,
                        "Unified Field": unified_col,
                    })
                
                if mapping_display:
                    st.dataframe(pd.DataFrame(mapping_display), use_container_width=True)
                else:
                    st.warning("Nincs automata felismerés :(")
            
            # Show unmapped columns
            if unmapped:
                unmapped_cols = st.expander(f"⚠️ Felismeretlen oszlopok ({len(unmapped)})")
                with unmapped_cols:
                    st.warning(f"A következő oszlopok nem kerültek besorolásra:")
                    for col in unmapped:
                        st.text(f"• {col}")
            
            # Preview
            st.subheader("📋 Adatok Előnézete (Raw)")
            st.dataframe(raw_df.head(5), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Hiba a fájl feltöltésekor: {str(e)}")

with tab2:
    if st.session_state.uploaded_data is not None:
        st.subheader("✅ Adatok Normalizálása & Validálása")
        
        try:
            # Normalize
            normalized_df = normalize_data(
                st.session_state.uploaded_data,
                st.session_state.mapping
            )
            st.session_state.normalized_data = normalized_df
            
            # Validate
            validation_issues = validate_data(normalized_df)
            
            if validation_issues:
                st.warning("### ⚠️ Validációs Figyelmeztetések")
                for issue in validation_issues:
                    st.warning(issue)
            else:
                st.success("### ✅ Minden OK! Az adatok készen állnak.")
            
            st.info(f"**Normalizált adatok**: {len(normalized_df)} sor × {len(normalized_df.columns)} oszlop")
        except Exception as e:
            st.error(f"❌ Hiba a normalizálás során: {str(e)}")
    else:
        st.info("Először töltsd fel az adatokat a '📥 Feltöltés & Mapping' fülön!")

with tab3:
    if st.session_state.normalized_data is not None:
        st.subheader("📊 Normalizált Adatok Előnézete")
        
        try:
            # Show statistics
            col1, col2, col3 = st.columns(3)
            
            df = st.session_state.normalized_data
            
            with col1:
                if "spend" in df.columns:
                    total_spend = df["spend"].sum()
                    st.metric("💰 Teljes Költség", f"{total_spend:,.0f} HUF")
            
            with col2:
                if "conversion_value" in df.columns:
                    total_value = df["conversion_value"].sum()
                    st.metric("💵 Konverziós Érték", f"{total_value:,.0f} HUF")
            
            with col3:
                if "roas" in df.columns:
                    avg_roas = df["roas"].mean()
                    st.metric("📈 Átlag ROAS", f"{avg_roas:.2f}")
            
            # Platform distribution
            if "platform" in df.columns:
                st.subheader("Platform Megoszlás")
                platform_dist = df["platform"].value_counts()
                st.bar_chart(platform_dist)
            
            # Data table
            st.subheader("Adatok Táblázat")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Hiba az előnézet során: {str(e)}")
    else:
        st.info("Először töltsd fel az adatokat a '📥 Feltöltés & Mapping' fülön!")

with tab4:
    if st.session_state.normalized_data is not None:
        st.subheader("💾 Adatok Exportálása")
        
        try:
            df = st.session_state.normalized_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 CSV letöltés",
                    data=csv,
                    file_name=f"hyper_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export as Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Campaigns')
                
                st.download_button(
                    label="📥 Excel letöltés",
                    data=buffer.getvalue(),
                    file_name=f"hyper_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"❌ Hiba az exportálás során: {str(e)}")
    else:
        st.info("Először töltsd fel az adatokat a '📥 Feltöltés & Mapping' fülön!")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**HYPER App v3.0** | Neuromarketing ROAS Predictor
- ✅ Fázis 1: CSV Importer & Intelligent Mapper
- ⏳ Fázis 2: Creative Analyzer (GPT4V)
- ⏳ Fázis 3: Live Channel Integration (API)
""")
