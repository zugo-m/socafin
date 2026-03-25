from ucimlrepo import fetch_ucirepo 
import pandas as pd
import os

'''# 1. Fetch dataset 
# Statlog (German Credit Data) ID is 144
german_credit = fetch_ucirepo(id=144) 
  
# 2. Extract features and targets as pandas dataframes 
X = german_credit.data.features 
y = german_credit.data.targets 

# 3. Combine into one dataframe for profiling
df = pd.concat([X, y], axis=1)

# 4. Save to your local data folder
os.makedirs('../data', exist_ok=True)
df.to_csv('../data/german_credit_raw.csv', index=False)

print("✅ Dataset successfully saved to data/german_credit_raw.csv")
print(f"Dataset shape: {df.shape}")'''

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def main():
    # 1. Dynamic Path Discovery
    # This finds the folder where THIS script lives (src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # This goes up one level to the project root (socafin)
    project_root = os.path.dirname(current_dir)
    
    # This defines the target data folder and file path
    data_folder = os.path.join(project_root, 'data')
    file_path = os.path.join(data_folder, 'german_credit_raw.csv')

    print(f"🚀 Starting data fetch...")

    # 2. Fetch dataset from UCI
    # Statlog (German Credit Data) ID is 144
    try:
        german_credit = fetch_ucirepo(id=144)
        
        # Extract features and targets
        X = german_credit.data.features
        y = german_credit.data.targets
        
        # Combine into one dataframe
        df = pd.concat([X, y], axis=1)

        # 3. Save to the correct location
        os.makedirs(data_folder, exist_ok=True)
        df.to_csv(file_path, index=False)

        print(f"✅ Success! File saved at: {file_path}")
        print(f"📊 Dataset Shape: {df.shape}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()