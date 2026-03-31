import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- SVG ASSET LIBRARY ---
#SVG_ICONS = {
 #   "finance": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>',
#   "chart": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
   # "profile": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>',
  #  "tools": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>',
  #  "comm": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg>',
  #  "logic": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>',
  # "lightbulb": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"></path><path d="M10 22h4"></path><path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"></path></svg>'
#}


# 1. Page Config & Professional Branding
st.set_page_config(page_title="SOCAFIN | AI Risk Intelligence", layout="wide")



# 2. Custom CSS
st.markdown("""
            
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8f9fc; }
    .stButton>button {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; border: none; padding: 0.75rem 2rem; border-radius: 8px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .report-card {
        background-color: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb; margin-bottom: 20px;
    }
    .metric-label { color: #6b7280; font-size: 0.875rem; font-weight: 500; }
    .metric-value { color: #111827; font-size: 1.5rem; font-weight: 800; }
    .sidebar-header { display: flex; align-items: center; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- DARK MODE TOGGLE LOGIC ---
with st.sidebar:
    st.markdown("---")
    dark_mode = st.checkbox("🌙 Enable Dark Mode")

# Define Professional Color Palettes
if dark_mode:
    bg_color = "#0f172a"      # Deep Navy/Slate
    card_bg = "#1e293b"       # Lighter Slate
    text_color = "#f8fafc"    # Off-white
    label_color = "#94a3b8"   # Muted Slate
    border_color = "#334155"
else:
    bg_color = "#f8f9fc"      # Light Gray/Blue
    card_bg = "#ffffff"       # Pure White
    text_color = "#111827"    # Near Black
    label_color = "#6b7280"   # Muted Gray
    border_color = "#e5e7eb"

# Inject Dynamic CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}

    /* 2. SPECIFIC FIX: Sidebar Background and Text */
    [data-testid="stSidebar"] {{
        background-color: {bg_color};
        border-right: 1px solid {border_color};
    }}
    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}

    /* 3. SPECIFIC FIX: Top Header/Toolbar Area */
    header[data-testid="stHeader"] {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    
    /* Report Card Styling */
    .report-card {{
        background-color: {card_bg};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid {border_color};
        margin-bottom: 20px;
    }}
    
    .metric-label {{ color: {label_color}; font-size: 0.875rem; font-weight: 500; }}
    .metric-value {{ color: {text_color}; font-size: 1.5rem; font-weight: 800; }}
    
    /* Adjusting standard Streamlit elements */
    h1, h2, h3, p {{ color: {text_color} !important; }}
    .stMarkdown {{ color: {text_color}; }}

    /* Fix for Widget Labels (Sliders, Selectboxes, etc) */
    .stSlider label, .stSelectbox label, .stNumberInput label {{
        color: {text_color} !important;
    }}
    
    /* 5. SPECIFIC FIX: Input Fields & Selectboxes in Dark Mode */
    
    /* Target the text inside the input boxes */
    div[data-baseweb="input"] input {{
        color: {text_color} !important;
        background-color: {card_bg} !important;
    }}

    /* Target the selectbox/dropdown container */
    div[data-baseweb="select"] > div {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
    }}

    /* Target the dropdown list items when they pop up */
    ul[role="listbox"] {{
        background-color: {card_bg} !important;
    }}
    
    ul[role="listbox"] li {{
        color: {text_color} !important;
    }}

    /* Target the number input increment/decrement buttons */
    div[data-testid="stNumberInput"] button {{
        background-color: {border_color} !important;
        color: {text_color} !important;
    }}

    /* Target the text color of the options in the dropdown */
    div[data-testid="stMarkdownContainer"] p {{
        color: {text_color};
    }}

    /* Button remains consistent with branding */
    .stButton>button {{
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; border: none; padding: 0.75rem 2rem; border-radius: 8px; font-weight: 600;
    }}

    /* Specific Card Colors */
    .text-approved {{ color: #10b981 !important; }} /* Green */
    .text-rejected {{ color: #ef4444 !important; }} /* Red */
    .text-risk {{ color: #f59e0b !important; }}      /* Amber/Gold */
    .text-reliability {{ color: #3b82f6 !important; }} /* Blue */
    </style>
    """, unsafe_allow_html=True)

# --- DYNAMIC SVG ASSET LIBRARY ---
# Note: stroke="{text_color}" allows the icon to flip between Navy and White automatically
SVG_ICONS = {
    "finance": f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>',
    
    "bank": f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M3 21h18"></path><path d="M3 10h18"></path><path d="M5 6l7-3 7 3"></path><path d="M4 10v11"></path><path d="M20 10v11"></path><path d="M8 14v3"></path><path d="M12 14v3"></path><path d="M16 14v3"></path></svg>',
    
    "chart": f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
    
    "profile": f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>',
    
    "tools": f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>',
    
    "comm": f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg>',
    
    "logic": f'<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>',
    
    "lightbulb": f'<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{text_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"></path><path d="M10 22h4"></path><path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"></path></svg>'
}

@st.cache_resource
def load_engine():
    model = joblib.load('models/credit_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    df_sample = pd.read_csv('data/german_credit_mapped.csv').drop('class', axis=1)
    feature_names = pd.get_dummies(df_sample).columns.tolist()
    return model, scaler, feature_names

model, scaler, feature_names = load_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f'<div class="sidebar-header">{SVG_ICONS["bank"]} <h2 style="margin:0; font-size:1.2rem;">Loan Officer Portal</h2></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. Core Financials
    st.markdown(f'{SVG_ICONS["finance"]} **Core Financials**', unsafe_allow_html=True)
    amount = st.number_input("Requested Amount (DM)", 100, 20000, 5000)
    duration = st.slider("Loan Duration (Months)", 1, 72, 24)
    installment = st.slider("Installment Rate (% of income)", 1, 4, 2)
    
    # 2. Account Quality
    st.markdown(f'<br>{SVG_ICONS["bank"]} **Account Quality**', unsafe_allow_html=True)
    checking = st.selectbox("Checking Status", ['< 0 DM', '0-200 DM', '>= 200 DM', 'no checking account'])
    savings = st.selectbox("Savings Status", ['< 100 DM', '100-500 DM', '500-1000 DM', '>= 1000 DM', 'no savings account'])
    
    # 3. Credit History
    st.markdown(f'<br>{SVG_ICONS["chart"]} **Credit Background**', unsafe_allow_html=True)
    history = st.selectbox("Credit History", ['no credits/all paid', 'all paid at this bank', 'existing paid until now', 'delay in paying past', 'critical account/other credits'])
    purpose = st.selectbox("Purpose", ['new car', 'used car', 'furniture/equipment', 'radio/television', 'domestic appliances', 'repairs', 'education', 'vacation', 'retraining', 'business', 'others'])
    existing_credits = st.slider("Existing Credits here", 1, 4, 1)

    # 4. Personal Profile
    st.markdown(f'<br>{SVG_ICONS["profile"]} **Personal Profile**', unsafe_allow_html=True)
    age = st.slider("Age", 18, 80, 30)
    sex_marital = st.selectbox("Sex & Marital Status", ['male : divorced/separated', 'female : divorced/separated/married', 'male : single', 'male : married/widowed', 'female : single'])
    residence_since = st.slider("Years at Current Residence", 1, 4, 2)
    housing = st.selectbox("Housing", ['own', 'rent', 'for free'])
    guarantors = st.selectbox("Other Debtors/Guarantors", ['none', 'co-applicant', 'guarantor'])
    num_dependents = st.number_input("Number of Dependents", 1, 2, 1)

    # 5. Employment & Assets
    st.markdown(f'<br>{SVG_ICONS["tools"]} **Professional Profile**', unsafe_allow_html=True)
    employment = st.selectbox("Employment Duration", ['unemployed', '< 1 year', '1-4 years', '4-7 years', '>= 7 years'])
    job = st.selectbox("Job Category", ['unskilled resident', 'unemployed non-resident', 'skilled', 'high qualif/self-emp/mgmt'])
    property_type = st.selectbox("Property Owned", ['real estate', 'life insurance', 'car/other', 'no property'])
    other_plans = st.selectbox("Other Installment Plans", ['bank', 'stores', 'none'])

    # 6. Legal & Comms
    st.markdown(f'<br>{SVG_ICONS["comm"]} **Legal & Comms**', unsafe_allow_html=True)
    telephone = st.selectbox("Telephone", ['none', 'yes, registered under customer name'])
    foreign = st.selectbox("Foreign Worker", ['yes', 'no'])

# --- MAIN DASHBOARD ---
st.markdown("<h1 style='color: #1e3a8a; margin-bottom: 0;'>SOCAFIN Risk Terminal</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>AI-Driven Interpretable Predictive Analytics for Banking Risk</p>", unsafe_allow_html=True)

if st.button("RUN ANALYTICS ENGINE 🚀", use_container_width=True):
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Feature Mapping
    input_df['duration'] = duration
    input_df['credit_amount'] = amount
    input_df['age'] = age
    input_df['installment_commitment'] = installment
    input_df['residence_since'] = residence_since
    input_df['existing_credits'] = existing_credits
    input_df['num_dependents'] = num_dependents
    
    selections = [
        ('checking_status', checking), ('savings_status', savings), 
        ('credit_history', history), ('purpose', purpose),
        ('employment', employment), ('personal_status', sex_marital),
        ('other_debtors', guarantors), ('property_magnitude', property_type),
        ('other_installment_plans', other_plans), ('housing', housing),
        ('job', job), ('own_telephone', telephone), ('foreign_worker', foreign)
    ]
    
    for col, val in selections:
        col_name = f"{col}_{val}"
        if col_name in feature_names:
            input_df[col_name] = 1

    cols_to_scale = ['duration', 'credit_amount', 'age', 'installment_commitment', 
                      'residence_since', 'existing_credits', 'num_dependents']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    prob = model.predict_proba(input_df)[0][1]
    prediction = "REJECTED (High Risk)" if prob > 0.5 else "APPROVED (Low Risk)"
    color = "#ef4444" if "REJECTED" in prediction else "#10b981"

    # KPI CARDS
    c1, c2, c3 = st.columns(3)
    # Decision Status Card
    with c1:
        status_class = "text-rejected" if "REJECTED" in prediction else "text-approved"
        st.markdown(f"""<div class='report-card'>
            <p class='metric-label'>DECISION STATUS</p>
            <p class='metric-value {status_class}'>{prediction}</p>
        </div>""", unsafe_allow_html=True)

    # Risk Probability Card
    with c2:
        st.markdown(f"""<div class='report-card'>
            <p class='metric-label'>RISK PROBABILITY</p>
            <p class='metric-value text-risk'>{prob:.2%}</p>
        </div>""", unsafe_allow_html=True)

    # Reliability Card
    with c3:
        st.markdown(f"""<div class='report-card'>
            <p class='metric-label'>MODEL RELIABILITY</p>
            <p class='metric-value text-reliability'>78.4% AUC</p>
        </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown(f'<div style="display:flex; align-items:center; gap:10px;">{SVG_ICONS["logic"]} <h3 style="margin:0;">Decision Logic (SHAP Waterfall)</h3></div>', unsafe_allow_html=True)
        explainer = shap.TreeExplainer(model)
        plt.style.use('dark_background' if dark_mode else 'default')
        shap_values = explainer(input_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        if dark_mode:
            fig.patch.set_facecolor(card_bg)
            ax.set_facecolor(card_bg)
        shap.plots.waterfall(shap_values[0,:,1], show=False)
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        st.pyplot(plt.gcf())
        
        st.markdown("---")
        st.markdown(f'<h3>Automated Risk Narrative</h3>', unsafe_allow_html=True)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'impact': shap_values.values[0,:,1]
        }).sort_values(by='impact', key=abs, ascending=False).head(3)

        st.write("Analysis results based on current weighted features:")
        for _, row in feature_importance.iterrows():
            tag = "INCREASED RISK" if row['impact'] > 0 else "DECREASED RISK"
            st.write(f"- **{row['feature']}**: {tag}")

    with col_right:
        st.markdown(f'<div style="display:flex; align-items:center; gap:10px;">{SVG_ICONS["lightbulb"]} <h3 style="margin:0;">Expert Advisor</h3></div>', unsafe_allow_html=True)
        if "REJECTED" in prediction:
            st.warning("Policy Recommendation: High probability of default. Consider requesting a higher collateral or reducing the loan duration.")
        else:
            st.success("Policy Recommendation: Applicant demonstrates strong liquidity. Standard processing recommended.")
        
        with st.expander("Technical Model Metrics"):
            st.write("Current Confusion Matrix Performance:")
            st.write("- Precision: 0.79 | Recall: 0.74")
            st.info("This model was cross-validated on the German Credit Dataset to minimize false approvals.")

else:
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h3 style='color: #64748b;'>Terminal Ready</h3>
            <p style='color: #94a3b8;'>Adjust the customer profile in the sidebar and execute the engine to generate the risk report.</p>
        </div>
    """, unsafe_allow_html=True)

# Place this at the very end of app.py
st.markdown("---")
with st.expander("🚀 Future Roadmap: LLM Integration"):
    st.info("""
    **V2.0 Planned Upgrade:** Transitioning the 'AI Risk Narrative' from a deterministic template to a **Local Small Language Model (SLM)**.
    - **Target Model:** FinGPT-1.1B or Phi-3 Mini.
    - **Deployment:** Ollama (Local Edge Inference).
    - **Goal:** Context-aware, conversational risk reports while maintaining 100% data privacy.
    """)