# Breast Cancer Prediction Web App

## 📌 Introduction

This project is a **Breast Cancer Prediction Web App** that utilizes **machine learning** to assist in diagnosing breast cancer. The model takes **29 important features** as input and predicts whether the tumor is **Benign (Non-Cancerous) or Malignant (Cancerous)**. Users can either **manually enter features** or **upload a medical report (PDF)** for automatic extraction.

## 🔍 Approach

- The project is built using **Flask** as the backend.
- **Machine Learning Model**: A trained model (`breast_cancer_model.pkl`) is used for prediction.
- **PDF Processing**: Extracts required features from uploaded PDF reports using `PyMuPDF (fitz)`.
- **Frontend**: Bootstrap and jQuery are used to provide a user-friendly interface.

## 💾 How to Download and Run the Project

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/prashant-g0/breast-cancer-detection-using-machine-learning.git
cd Breast-Cancer-app
```

### 2️⃣ Install Required Libraries

Before running the project, install the necessary dependencies:

```sh
pip install -r requirements.txt
```
OR
## 📦 Pre-Downloads (Required Libraries)

Ensure you have the following libraries installed:

```sh
pip install flask joblib numpy fitz opencv-python pandas
```

### 3️⃣ Run the Flask App

```sh
python app.py
```

### 4️⃣ Open in Browser

Go to: [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.


## 👨‍💻 Author

**Prashant Gupta**\
🔗 [LinkedIn](https://www.linkedin.com/in/prashant-g0)\
📷 [Instagram](https://www.instagram.com/prashantg.0)

---
