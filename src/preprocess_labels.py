import pandas as pd
import os

def decode_german_credit():
    # 1. Path Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = os.path.join(project_root, 'data', 'german_credit_raw.csv')
    output_path = os.path.join(project_root, 'data', 'german_credit_mapped.csv')

    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Run fetch_data.py first.")
        return

    # 2. Load Raw Data
    df = pd.read_csv(input_path)

    # 3. Full UCI Mapping (Verified Metadata)
    mappings = {
        'checking_status': {'A11': '< 0 DM', 'A12': '0 <= x < 200 DM', 'A13': '>= 200 DM', 'A14': 'no checking account'},
        'credit_history': {'A30': 'no credits/all paid', 'A31': 'all paid back here', 'A32': 'existing paid back', 'A33': 'delay in past', 'A34': 'critical account'},
        'purpose': {'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equip', 'A43': 'radio/tv', 'A44': 'domestic appliances', 'A45': 'repairs', 'A46': 'education', 'A47': 'vacation', 'A48': 'retraining', 'A49': 'business', 'A410': 'others'},
        'savings_status': {'A61': '< 100 DM', 'A62': '100 <= x < 500 DM', 'A63': '500 <= x < 1000 DM', 'A64': '>= 1000 DM', 'A65': 'no savings'},
        'employment': {'A71': 'unemployed', 'A72': '< 1 yr', 'A73': '1 <= x < 4 yrs', 'A74': '4 <= x < 7 yrs', 'A75': '>= 7 yrs'},
        'personal_status': {'A91': 'male:divorced', 'A92': 'female:div/dep/mar', 'A93': 'male:single', 'A94': 'male:mar/wid', 'A95': 'female:single'},
        'other_parties': {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'},
        'property_magnitude': {'A121': 'real estate', 'A122': 'life insurance', 'A123': 'car', 'A124': 'no property'},
        'other_payment_plans': {'A141': 'bank', 'A142': 'stores', 'A143': 'none'},
        'housing': {'A151': 'rent', 'A152': 'own', 'A153': 'for free'},
        'job': {'A171': 'unemp/unskilled non-res', 'A172': 'unskilled resident', 'A173': 'skilled', 'A174': 'high qual/mgmt'},
        'own_telephone': {'A191': 'none', 'A192': 'yes'},
        'foreign_worker': {'A201': 'yes', 'A202': 'no'},
        'class': {1: 'Good Risk', 2: 'Bad Risk'} # Mapping the target variable
    }

    # 4. Apply the Transformation
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])

    # 5. Save
    df.to_csv(output_path, index=False)
    print(f"✅ Readable data successfully saved to: {output_path}")

if __name__ == "__main__":
    decode_german_credit()