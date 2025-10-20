import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(num_customers=5000):
    np.random.seed(42)
    
    customers = []
    for i in range(num_customers):
        customer = {
            'customer_id': f'CUST_{i:05d}',
            'tenure_days': np.random.randint(1, 365*3),
            'monthly_charges': np.random.normal(65, 20),
            'total_charges': np.random.normal(1500, 800),
            'contract_type': np.random.choice(['Monthly', 'Annual', 'Two-Year'], p=[0.6, 0.3, 0.1]),
            'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check']),
            'paperless_billing': np.random.choice([0, 1]),
            'support_calls_3mo': np.random.poisson(2),
            'feature_usage_score': np.random.normal(0.7, 0.2),
            'satisfaction_score': np.random.normal(3.5, 1.0),
            'days_since_last_upgrade': np.random.randint(0, 365),
            'churn_risk': np.random.uniform(0, 1)
        }
        customers.append(customer)
    
    df = pd.DataFrame(customers)
    
    # Create churn labels based on risk factors
    df['churned'] = (
        (df['support_calls_3mo'] > 4) |
        (df['satisfaction_score'] < 2.5) |
        (df['tenure_days'] < 30) |
        (df['monthly_charges'] > df['monthly_charges'].quantile(0.8))
    ).astype(int)
    
    df.to_csv('data/customer_data.csv', index=False)
    print(f"Generated data for {len(df)} customers")
    return df

if __name__ == "__main__":
    generate_customer_data()