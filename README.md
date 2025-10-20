ECHO is on.
# ?? Customer Churn Prediction System 
  
A machine learning system that predicts customer churn, diagnoses root causes, and recommends targeted retention strategies using real telecom industry data. 
  
## ?? Features  
  
- **Real-time Churn Prediction**: XGBoost model with 79% accuracy 
- **Context-Aware Diagnostics**: Identifies why customers might leave  
- **Smart Interventions**: Personalized retention strategies  
- **Interactive Dashboard**: Streamlit web interface  
- **Real Industry Data**: IBM Telco Customer Churn dataset 
  
## ??? Tech Stack  
  
- **Machine Learning**: XGBoost, Scikit-learn, SHAP  
- **Backend**: Python, Pandas, NumPy  
- **Frontend**: Streamlit, Plotly  
- **Data**: IBM Telco Dataset (7,043 customers) 
  
## ?? Quick Start  
  
\`\`\`bash  
# Clone repository  
git clone https://github.com/VaradCodes31/customer-churn-prediction.git  
  
# Setup environment  
python -m venv churn_venv  
churn_venv\Scripts\activate  
pip install -r requirements.txt  
  
# Run dashboard  
streamlit run dashboard/app.py  
\`\`\` 
  
## ?? Model Performance  
  
- **Accuracy**: 79%  
- **ROC-AUC**: 0.837  
- **Precision/Recall**: Balanced for business use 
  
## ?? Business Impact  
  
- Reduces customer acquisition costs 5-7x  
- Enables proactive customer retention  
- Provides actionable insights for business teams 
  
## ?? Project Structure  
  
\`\`\`  
customer-churn-prediction/  
ÃÄÄ dashboard/          # Streamlit web app  
  
## ?? Project Structure  
  
```  
customer-churn-prediction/  
|-- dashboard/          # Streamlit web app  
  
## Project Structure  
  
- dashboard/ - Streamlit web app  
- models/ - ML models and intervention engine  
- data/ - Data processing scripts 
