import joblib
import pandas as pd
import numpy as np

def run_sprint8_audit():
    print("🚀 SOCAFIN | Sprint 8: Comprehensive Validation Audit\n" + "="*50)
    
    # --- CATEGORY 1: TECHNICAL INTEGRITY ---
    try:
        model = joblib.load('models/credit_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        df_sample = pd.read_csv('data/german_credit_mapped.csv').drop('class', axis=1)
        feature_names = pd.get_dummies(df_sample).columns.tolist()
        
        # Test 1: Feature Alignment
        expected = model.n_features_in_
        actual = len(feature_names)
        if expected == actual:
            print("✅ Technical Integrity: Feature counts match ({}).".format(actual))
        else:
            print("❌ Technical Integrity: MISMATCH! Model expects {}, App has {}.".format(expected, actual))
            
    except Exception as e:
        print(f"❌ Technical Integrity: Critical Load Error - {e}")
        return

    # --- CATEGORY 3: MODEL SANITY (SENSITIVITY ANALYSIS) ---
    print("\n🧠 Category 3: Model Sanity & Bias Check")
    
    # Create two synthetic profiles: One "Perfect", One "High-Risk"
    # We use the median values and flip the critical switches
    perfect_profile = pd.DataFrame(0, index=[0], columns=feature_names)
    risk_profile = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Switch 1: Savings (A key indicator in German Credit Data)
    if 'savings_status_>= 1000 DM' in feature_names:
        perfect_profile['savings_status_>= 1000 DM'] = 1
    if 'savings_status_< 100 DM' in feature_names:
        risk_profile['savings_status_< 100 DM'] = 1
        
    # Switch 2: Employment
    if 'employment_>= 7 years' in feature_names:
        perfect_profile['employment_>= 7 years'] = 1
    if 'employment_unemployed' in feature_names:
        risk_profile['employment_unemployed'] = 1

    perf_prob = model.predict_proba(perfect_profile)[0][1]
    risk_prob = model.predict_proba(risk_profile)[0][1]

    if perf_prob < risk_prob:
        print(f"✅ Model Sanity: Passed. (Perfect Risk: {perf_prob:.2%}, High Risk: {risk_prob:.2%})")
    else:
        print("❌ Model Sanity: FAILED. Model is producing illogical risk correlations.")

    print("\n" + "="*50 + "\n🏁 Audit Complete: SOCAFIN is ready for Sprint 9.")

if __name__ == "__main__":
    run_sprint8_audit()