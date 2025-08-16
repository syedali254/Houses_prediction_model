
# 🏠 House Price Prediction with Streamlit

This project is a **Machine Learning + Streamlit web app** that predicts house prices based on features such as area, number of bedrooms, bathrooms, and other parameters. The application trains a model and directly serves predictions through an interactive Streamlit interface.

---

## 📌 Features

* 📊 **Data Preprocessing** – Cleans and prepares the dataset.
* 🤖 **Model Training** – Trains a regression model (Linear Regression by default).
* 💾 **Single File Application** – No external `.pkl` needed, the model is trained inside the app.
* 🌐 **Interactive Streamlit UI** – Enter house details and get instant price predictions.
* 🚀 **Deployable on Streamlit Cloud** – Easy hosting without large files.

---

## 🛠️ Tech Stack

* **Python 3.x**
* **pandas, numpy** – Data handling
* **scikit-learn** – Machine Learning model
* **Streamlit** – Web app frontend

---

## 📂 Project Structure

```
house_price_app.py    # Single file containing model training + Streamlit app
README.md             # Documentation
requirements.txt      # Python dependencies
```

---

## ⚡ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run house_price_app.py
```

---

## 🎯 How It Works

1. The app loads the dataset (or synthetic data).
2. Preprocessing is applied (handling missing values, scaling if required).
3. A **Linear Regression model** is trained each time the app starts.
4. The Streamlit interface allows the user to input house details.
5. The trained model predicts the **estimated price** in real-time.

---

## 🖼️ Example UI

* **Input fields:** Area, Bedrooms, Bathrooms, Location (optional).
* **Output:** Predicted House Price.

---

## 🚀 Deployment on Streamlit Cloud

1. Push code to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your repo and select `house_price_app.py`.
4. App will be live at:

   ```
   https://your-username-house-price.streamlit.app
   ```

---

## ✅ Future Improvements

* Add more advanced models (XGBoost, Random Forest).
* Improve dataset with real housing data.
* Save trained model externally to avoid retraining.
* Enhance UI with charts & visualizations.

---

## 👨‍💻 Author

Developed by **Muhammad Ali**

---
