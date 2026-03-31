# SOCAFIN: Interpretable AI Credit Risk Terminal
**Project Author:** Marvelous Ojukwu  
**Technical Domain:** Data Engineering & Explainable AI (XAI)  
**Version:** 1.0-Stable | **Build Date:** March 2026

---

## 1. Executive Overview
SOCAFIN (Sovereign Capital Financial Intelligence) is a professional-grade Decision Support System (DSS) developed to quantify credit default risk. Built upon the **Statlog German Credit Dataset**, the terminal bridges the gap between high-performance machine learning and **Explainable AI (XAI)**. By integrating SHAP (SHapley Additive exPlanations), SOCAFIN provides a transparent, feature-level breakdown of every loan decision, ensuring compliance with financial "Right to Explanation" standards and mitigating "black-box" algorithmic bias.



## 2. The 9-Sprint Journey: Development Lifecycle
The project was executed through a modular, sprint-based architecture to ensure technical rigor and iterative validation.

### 🟢 Phase I: Data Engineering (Sprints 1-3)
* **Sprint 1: Data Ingestion & ETL** — Extracted raw UCI Statlog data and mapped cryptic categorical encodings into human-readable ordinals.
* **Sprint 2: Preprocessing & Vectorization** — Implemented One-Hot Encoding for multi-class features, expanding the feature space to 61 dimensions.
* **Sprint 3: Robust Scaling** — Applied `RobustScaler` to numerical features (`Amount`, `Age`, `Duration`) to neutralize the influence of financial outliers.

### 🔵 Phase II: Intelligence Engine (Sprints 4-6)
* **Sprint 4: Ensemble Modelling** — Developed a **Random Forest Classifier** optimized via cross-validation.
* **Sprint 5: Performance Auditing** — Validated the model with a **78.4% AUC-ROC**, focusing on the Precision-Recall trade-off for high-risk lending.
* **Sprint 6: XAI Integration** — Deployed **SHAP TreeExplainer** to calculate marginal feature contributions for individual and global predictions.



### 🟠 Phase III: Production & UI (Sprints 7-9)
* **Sprint 7: Terminal Engineering** — Built the **Streamlit** dashboard with segmented input logic (Financials, Quality, Profile).
* **Sprint 8: System Hardening** — Implemented Professional Dark Mode, SVG asset synchronization, and input boundary guards.
* **Sprint 9: Deployment & Documentation** — Finalized environment pinning and Master's-level technical documentation.

---

## 3. Technical Architecture
### Core Stack
* **Language:** Python 3.9+
* **ML Framework:** Scikit-learn (Random Forest Ensemble)
* **Interpretability:** SHAP (Shapley Additive exPlanations)
* **UI Framework:** Streamlit (Custom CSS/Inter-font)

### Performance Metrics
* **AUC-ROC:** 0.784
* **Precision (Approved):** 0.79
* **Recall (Approved):** 0.74

---

## 4. Complete Directory Structure
socafin/  
├── .venv/                     # Python Virtual Environment  
├── app.py                     # Main Production Application Logic  
├── requirements.txt           # Pinned Environment Dependencies  
├── README.md                  # Technical Whitepaper  
├── .gitignore                 # Version Control Exclusion Rules  
├── validate_sprint8.py        # TDD/QA Integrity Script  
│  
├── data/                      # Data Assets  
│   ├── german_credit_mapped.csv  
│   └── german_credit_raw.csv  
│  
├── models/                    # Serialized ML Artifacts  
│   ├── credit_model.pkl       
│   └── scaler.pkl             
│  
├── notebooks/                 # R&D Research Trail  
│   ├── 01_data_profiling.ipynb  
│   ├── 02_preprocessing.ipynb  
│   ├── 03_feature_selection.ipynb  
│   ├── 04_model_training.ipynb  
│   ├── 05_model_evaluation.ipynb  
│   └── 06_interpretability.ipynb  
│  
├── src/                       # Modular Utility Scripts  
│   ├── fetch_data.py  
│   └── preprocess_labels.py  
│  
└── docs/                      # Supplemental Documentation  

---

## 5. Installation & Execution
To replicate this environment with precision:

1. **Clone & Navigate:**
   git clone https://github.com/yourusername/socafin.git
   cd socafin

2. **Environment Setup:**
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

3. **Dependency Installation:**
   pip install -r requirements.txt

4. **Launch Terminal:**
   streamlit run app.py

---

## 6. Future Roadmap: SOCAFIN V2.0
* **Local SLM Integration:** Transitioning from deterministic narratives to **Ollama-powered** Small Language Models (e.g., Phi-3) for conversational auditing.
* **Real-time Bias Monitoring:** Implementing automated fairness checks to ensure equitable lending across demographic subsets.

---
© 2026 Marvelous Ojukwu. Developed for Advanced AI/ML Engineering Portfolio.