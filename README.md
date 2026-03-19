# 🚀 **Customer Churn Prediction System**

---

## 🌟 **Overview**

An end-to-end **Machine Learning project** designed to predict customer churn using a full-stack architecture.
This project demonstrates the complete data science lifecycle—from data preprocessing and model training to deployment via API and an interactive dashboard.

---

## 🎯 **Key Features**

* 🔍 **Churn Prediction Model** using classification algorithms
* ⚙️ **REST API** built with FastAPI for real-time inference
* 🌐 **Interactive Dashboard** using Streamlit
* 📊 **Feature Importance Visualization**
* 📉 **SHAP-based Model Explainability**
* 🎛 **Dynamic Probability Threshold Slider**
* 📈 **Performance Metrics (Accuracy & ROC-AUC)**

---

## 🧠 **Machine Learning Workflow**

1. Data Cleaning & Preprocessing
2. Feature Engineering (TotalSpend, AvgMonthlyValue, HighValue)
3. Model Training (Random Forest + GridSearchCV)
4. Model Evaluation (Accuracy, ROC-AUC, Classification Report)
5. Model Deployment via API
6. Real-time Predictions via UI

---

## 🛠 **Tech Stack**

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Explainability:** SHAP
* **Model Storage:** Joblib

---

## 📊 **Model Performance**

* ✅ **Accuracy:** ~80%
* 📈 **ROC-AUC:** ~0.84
* 📉 Balanced predictions using class weighting

---

## 📁 **Project Structure**

```
customer-churn-ml/
│── data/
│── src/
│── api/
│── app/
│── models/
│── requirements.txt
```

---

## ▶️ **How to Run Locally**

```bash
pip install -r requirements.txt

python src/train.py
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

---

## 🌐 **API Endpoint**

* **POST /predict**
* Input: Customer details (JSON)
* Output: Prediction + Probability

---

## 📊 **Dashboard Features**

* Interactive sliders for input features
* Real-time churn prediction
* Adjustable decision threshold
* Feature importance visualization
* Model performance display

---

## 💡 **Business Use Case**

Helps businesses:

* Identify customers at risk of churn
* Take proactive retention actions
* Improve customer lifetime value

---

## 🚀 **Future Improvements**

* Cloud Deployment (AWS / Render)
* Dockerization 🐳
* Model Monitoring & Logging
* CI/CD Pipeline
* User Authentication

---

## 👨‍💻 **Author**

**Priyanu Sikdar**

---

## ⭐ **GitHub**

If you found this project useful, consider giving it a ⭐

---
