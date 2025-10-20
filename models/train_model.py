import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import shap

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        
    def preprocess_data(self, df):
        """Prepare data for training"""
        # Encode categorical variables
        categorical_cols = ['contract_type', 'payment_method']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Select features for Telco dataset
        feature_cols = ['tenure_days', 'monthly_charges', 'total_charges', 'contract_type',
                       'payment_method', 'senior_citizen', 'support_calls_3mo',
                       'feature_usage_score', 'satisfaction_score', 'tech_support']
        
        self.feature_names = feature_cols
        X = df[feature_cols]
        y = df['churned']
        
        return X, y
    
    def train(self, df):
        """Train the churn prediction model"""
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            base_score=0.5
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        return self.model
    
    def predict_churn_risk(self, customer_data):
        """Predict churn risk for individual customer"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare single customer data
        customer_df = pd.DataFrame([customer_data])
        
        # Encode categorical variables - handle unseen labels safely
        for col in ['contract_type', 'payment_method']:
            if col in customer_df.columns:
                # Get the original values seen during training
                known_labels = list(self.label_encoders[col].classes_)
                current_value = str(customer_df[col].iloc[0])  # Ensure string type
                
                # If value wasn't seen during training, use a default
                if current_value not in known_labels:
                    # Use the first label from training as default
                    customer_df[col] = self.label_encoders[col].transform([known_labels[0]])[0]
                else:
                    customer_df[col] = self.label_encoders[col].transform([current_value])[0]
        
        # Ensure all features are present and convert to numeric
        for feature in self.feature_names:
            if feature not in customer_df.columns:
                customer_df[feature] = 0
            else:
                # Ensure numeric type
                customer_df[feature] = pd.to_numeric(customer_df[feature], errors='coerce')
                customer_df[feature].fillna(0, inplace=True)
        
        customer_df = customer_df[self.feature_names]
        
        # Get prediction and probabilities
        risk_score = self.model.predict_proba(customer_df)[0, 1]
        prediction = self.model.predict(customer_df)[0]
        
        return risk_score, prediction
    
    def explain_prediction(self, customer_data):
        """Use SHAP to explain why a customer might churn"""
        # Prepare data
        customer_df = pd.DataFrame([customer_data])
        for col in ['contract_type', 'payment_method']:
            if col in customer_df.columns:
                customer_df[col] = self.label_encoders[col].transform(
                    customer_df[col].astype(str)
                )
        
        for feature in self.feature_names:
            if feature not in customer_df.columns:
                customer_df[feature] = 0
        
        customer_df = customer_df[self.feature_names]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(customer_df)
        
        # Get feature importance for this prediction
        feature_importance = dict(zip(self.feature_names, shap_values[0]))
        
        return feature_importance
    
    def save_model(self, path='models/churn_model.pkl'):
        """Save trained model and encoders"""
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
    
    def load_model(self, path='models/churn_model.pkl'):
        """Load trained model and encoders with error handling"""
        try:
            saved_data = joblib.load(path)
            self.model = saved_data['model']
            self.label_encoders = saved_data['label_encoders']
            self.feature_names = saved_data['feature_names']
        except FileNotFoundError:
            # Model doesn't exist yet
            raise FileNotFoundError(f"Model file {path} not found. Train the model first.")

# Train the model
if __name__ == "__main__":
    # Load REAL Telco data
    df = pd.read_csv('data/telco_processed.csv')
    
    # Train model
    predictor = ChurnPredictor()
    predictor.train(df)
    predictor.save_model()
    
    print("Model trained and saved successfully!")