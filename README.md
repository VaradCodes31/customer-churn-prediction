# 🧠 Customer Churn Prediction & Retention System

A **full-stack machine learning project** that predicts customer churn, explains the key reasons behind it, and recommends targeted retention actions. Built using the **IBM Telco Customer Churn dataset**, the system achieves **79% accuracy** and is deployed via an **interactive Streamlit dashboard** designed for business decision-makers.

---

## 🚀 Project Overview

Customer churn — the loss of customers over time — costs the telecom industry **over $25 billion annually**.
This project aims to help businesses **identify at-risk customers early**, understand **why they’re leaving**, and **take data-driven actions** to retain them.

The solution integrates **machine learning, explainable AI (XAI)**, and **data visualization** to deliver **actionable insights** in a production-ready, user-friendly web interface.

---

## 🧩 Features

* ✅ **Churn Prediction Model** — Classifies customers as likely to churn or stay with 79% accuracy.
* 🔍 **Explainable AI Insights** — Uses **SHAP** to reveal feature importance and explain individual churn predictions.
* 💡 **Retention Recommendation Engine** — Suggests targeted strategies to retain at-risk customers.
* 📊 **Interactive Streamlit Dashboard** — Visualizes customer churn risk scores and explanations dynamically.
* ⚙️ **End-to-End System Design** — Covers data acquisition, preprocessing, model training, and deployment.

---

## 🧠 Tech Stack

| Category            | Tools & Technologies                                   |
| ------------------- | ------------------------------------------------------ |
| **Programming**     | Python                                                 |
| **Libraries**       | Pandas, NumPy, Scikit-learn, SHAP, Matplotlib, Seaborn |
| **Frameworks**      | Streamlit                                              |
| **Database**        | SQL (optional integration for data storage)            |
| **Version Control** | Git, GitHub                                            |
| **Deployment**      | Streamlit Cloud / Local Host                           |

---

## 📊 Data

**Dataset:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

* **Rows:** 7,043 customers
* **Features:** 21 (Demographics, Contract Type, Internet Services, etc.)
* **Target Variable:** `Churn` (Yes/No)

---

## ⚙️ Project Workflow

1. **Data Collection & Cleaning**

   * Loaded and preprocessed data (handled missing values, categorical encoding, feature scaling).

2. **Exploratory Data Analysis (EDA)**

   * Identified churn patterns by demographics, service type, and contract duration.
   * Visualized churn drivers using Seaborn and Matplotlib.

3. **Model Development**

   * Trained multiple classifiers (Logistic Regression, Random Forest, XGBoost).
   * Tuned hyperparameters and selected the best-performing model based on F1-score.

4. **Explainability with SHAP**

   * Implemented SHAP summary and dependence plots to interpret feature influence.

5. **Retention Recommendation Engine**

   * Created rule-based suggestions (e.g., offer discounts, upgrade plans, or improve customer service).

6. **Dashboard Deployment**

   * Built a **Streamlit dashboard** integrating model predictions, risk visualization, and SHAP insights.

---

## 💻 How to Run Locally

1. **Clone this repository**

   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**

   * Open the local URL displayed in the terminal (usually `http://localhost:8501`).

---

## 🧾 Results

| Metric                 | Score |
| ---------------------- | ----- |
| Accuracy               | 79%   |
| F1-Score (Churn class) | 0.81  |
| Precision              | 0.77  |
| Recall                 | 0.79  |

**Key Insights:**

* Customers on month-to-month contracts with high monthly charges are most likely to churn.
* Long-term contracts and bundled services significantly reduce churn risk.

---

## 🧭 Future Improvements

* Integrate **real-time data pipelines** (Kafka / REST API).
* Deploy on **AWS / GCP with CI/CD integration**.
* Add **automated retraining** for model drift management.
* Implement **A/B testing** for retention strategies.

---

## 👤 Author

**Varad Alshi**
📍 Pune, India
📧 [varadalshi7@gmail.com](mailto:varadalshi7@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/varadalshi) | [GitHub](https://github.com/VaradCodes31)

---

## ⭐ Acknowledgments

Dataset provided by IBM Telco Customer Churn.
Inspired by real-world telecom analytics use cases focusing on **customer retention and data-driven business strategy**.
