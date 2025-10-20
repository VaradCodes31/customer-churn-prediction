import pandas as pd
import numpy as np

def load_and_preprocess_telco_data():
    """Load and preprocess the real Telco customer churn dataset"""
    
    # Load the dataset
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create churn label (convert 'Yes'/'No' to 1/0)
    df['churned'] = (df['Churn'] == 'Yes').astype(int)
    
    # Handle TotalCharges - it has some empty strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Select and rename key features for our model
    processed_df = pd.DataFrame()
    
    # Customer identifiers
    processed_df['customer_id'] = df['customerID']
    
    # Core features (mapping to our existing feature names)
    processed_df['tenure_days'] = df['tenure'] * 30  # Convert months to approximate days
    processed_df['monthly_charges'] = df['MonthlyCharges']
    processed_df['total_charges'] = df['TotalCharges']
    
    # Contract information
    processed_df['contract_type'] = df['Contract']
    
    # Payment method
    processed_df['payment_method'] = df['PaymentMethod']
    
    # Service information
    processed_df['internet_service'] = df['InternetService']
    processed_df['phone_service'] = (df['PhoneService'] == 'Yes').astype(int)
    
    # Usage patterns (simplified)
    processed_df['multiple_lines'] = (df['MultipleLines'] == 'Yes').astype(int)
    processed_df['online_security'] = (df['OnlineSecurity'] == 'Yes').astype(int)
    processed_df['tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
    
    # Demographics
    processed_df['gender'] = df['gender']
    processed_df['senior_citizen'] = df['SeniorCitizen']
    processed_df['partner'] = (df['Partner'] == 'Yes').astype(int)
    processed_df['dependents'] = (df['Dependents'] == 'Yes').astype(int)
    
    # Create synthetic features similar to our original dataset
    processed_df['support_calls_3mo'] = np.random.poisson(2, len(df))  # Simulated support calls
    processed_df['feature_usage_score'] = np.random.normal(0.7, 0.2, len(df))  # Simulated usage score
    processed_df['satisfaction_score'] = np.random.normal(3.5, 1.0, len(df))  # Simulated satisfaction
    processed_df['days_since_last_upgrade'] = np.random.randint(0, 365, len(df))  # Simulated upgrade date
    
    # Churn label
    processed_df['churned'] = df['churned']
    
    # Calculate churn risk based on actual churn rate and key factors
    churn_rate = df['churned'].mean()
    base_risk = churn_rate
    
    # Enhance risk based on known churn factors
    risk_factors = (
        (df['Contract'] == 'Month-to-month').astype(int) * 0.3 +
        (df['tenure'] < 12).astype(int) * 0.2 +
        (df['MonthlyCharges'] > 70).astype(int) * 0.2 +
        (df['SeniorCitizen'] == 1).astype(int) * 0.1 +
        (df['OnlineSecurity'] == 'No').astype(int) * 0.1 +
        (df['TechSupport'] == 'No').astype(int) * 0.1
    )
    
    processed_df['churn_risk'] = np.clip(base_risk + risk_factors * 0.5, 0, 1)
    
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Churn rate: {processed_df['churned'].mean():.2%}")
    print(f"Average churn risk: {processed_df['churn_risk'].mean():.2%}")
    
    # Save the processed data
    processed_df.to_csv('data/telco_processed.csv', index=False)
    print("Processed Telco data saved to 'data/telco_processed.csv'")
    
    return processed_df

if __name__ == "__main__":
    df = load_and_preprocess_telco_data()
    print("\nFirst few rows of processed data:")
    print(df[['customer_id', 'tenure_days', 'monthly_charges', 'churned', 'churn_risk']].head())