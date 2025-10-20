import pandas as pd
import numpy as np
from train_model import ChurnPredictor

class InterventionEngine:
    def __init__(self):
        self.churn_predictor = ChurnPredictor()
        self.churn_predictor.load_model()
    
    def diagnose_churn_reason(self, customer_data, shap_explanation=None):
        """Identify primary reason for churn risk using Telco features"""
        reasons = []
        
        # Price sensitivity - Telco specific
        if customer_data.get('monthly_charges', 0) > 70:
            reasons.append(("Price Sensitivity", "High monthly charges", 0.8))
        
        # Contract type - Major churn driver in Telco
        if customer_data.get('contract_type') == 'Month-to-month':
            reasons.append(("Contract Flexibility", "Month-to-month contract (high risk)", 0.9))
        
        # Tenure - New customers more likely to churn
        if customer_data.get('tenure_days', 0) < 180:  # Less than 6 months
            reasons.append(("New Customer", "Low tenure (high churn risk)", 0.7))
        
        # Senior citizens - Higher churn rate
        if customer_data.get('senior_citizen', 0) == 1:
            reasons.append(("Senior Citizen", "Higher risk demographic", 0.6))
        
        # Missing tech support - Service quality issue
        if customer_data.get('tech_support', 1) == 0:
            reasons.append(("Service Quality", "No tech support service", 0.5))
        
        # Support calls - Service issues
        if customer_data.get('support_calls_3mo', 0) >= 3:
            reasons.append(("Support Issues", "Multiple support calls", 0.7))
        
        # Low feature usage
        if customer_data.get('feature_usage_score', 1) < 0.4:
            reasons.append(("Low Engagement", "Low service usage", 0.6))
        
        # Sort by impact score
        reasons.sort(key=lambda x: x[2], reverse=True)
        
        return [{"reason": r, "description": d, "impact": i} for r, d, i in reasons]
    
    def recommend_intervention(self, churn_risk, primary_reason, customer_data):
        """Recommend specific intervention based on churn reason"""
        risk_level = "high" if churn_risk > 0.7 else "medium" if churn_risk > 0.4 else "low"
        reason = primary_reason["reason"]
        
        interventions = {
            "Price Sensitivity": {
                "low": "Send personalized email with cost-saving tips",
                "medium": "Offer 15% loyalty discount for 6 months",
                "high": "Assign account manager for personalized pricing review"
            },
            "Support Issues": {
                "low": "Proactive check-in from support team",
                "medium": "Dedicated support specialist assignment",
                "high": "Executive escalation and service credit offer"
            },
            "Low Engagement": {
                "low": "Send feature tutorial emails",
                "medium": "Schedule onboarding call with product specialist",
                "high": "Offer personalized training session"
            },
            "Contract Flexibility": {
                "low": "Educate about annual plan benefits",
                "medium": "Offer incentive to switch to annual plan",
                "high": "Create custom contract with flexibility options"
            },
            "Low Satisfaction": {
                "low": "Request feedback and show appreciation",
                "medium": "Personalized recovery campaign",
                "high": "Executive apology and service recovery package"
            },
            "New Customer": {
                "low": "Welcome call and onboarding support",
                "medium": "Dedicated new customer success manager",
                "high": "Extended trial period and personalized setup"
            },
            "Senior Citizen": {
                "low": "Senior-friendly communication materials",
                "medium": "Dedicated senior support line",
                "high": "Personalized senior discount program"
            },
            "Service Quality": {
                "low": "Proactive service quality check",
                "medium": "Free service upgrade offer",
                "high": "Executive service review and compensation"
            }
        }
        
        intervention = interventions.get(reason, {}).get(risk_level, "Monitor customer behavior")
        
        return {
            "risk_level": risk_level,
            "primary_reason": reason,
            "intervention": intervention,
            "urgency": "immediate" if churn_risk > 0.7 else "within_week" if churn_risk > 0.4 else "monitor"
        }

# Test the system
if __name__ == "__main__":
    engine = InterventionEngine()
    
    # Load sample customer from Telco data
    df = pd.read_csv('data/telco_processed.csv')
    sample_customer = df.iloc[0].to_dict()
    
    # Get prediction
    risk_score, prediction = engine.churn_predictor.predict_churn_risk(sample_customer)
    
    # Diagnose and recommend (without SHAP for now)
    reasons = engine.diagnose_churn_reason(sample_customer)
    if reasons:
        intervention = engine.recommend_intervention(risk_score, reasons[0], sample_customer)
        print(f"Churn Risk: {risk_score:.3f}")
        print(f"Primary Reason: {reasons[0]['reason']}")
        print(f"Intervention: {intervention['intervention']}")