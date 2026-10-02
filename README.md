# 🛡️ Spam Message Detector

An AI-powered Spam Message Detector built using Python, Machine Learning, Flask, and Scikit-learn.

## 🚀 Features

- Detects whether a message is Spam or Not Spam
- Shows prediction confidence
- Simple and responsive web interface
- Machine Learning based classification
- Fast message prediction

## 🧠 Machine Learning

The project uses:

- TF-IDF Vectorization
- Logistic Regression
- Scikit-learn Pipeline

The model is trained using the SMS Spam Collection dataset.

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML
- CSS

## 📁 Project Structure

```text
spam-message-detector/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── SMSSpamCollection
├── model/
│   └── spam_model.pkl
├── static/
│   └── style.css
└── templates/
    └── index.html