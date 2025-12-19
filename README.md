🌿 Plant Disease Detection System using Deep Learning
📌 Project Description

This project is a Deep Learning–based Plant Disease Detection System that identifies plant leaf diseases from images using a trained Convolutional Neural Network (CNN) model. The application is built with Python, TensorFlow/Keras, OpenCV, and Streamlit, providing an easy-to-use web interface for real-time disease prediction.

Users can upload a plant leaf image, and the system classifies it as healthy or diseased, along with a confidence score and prevention suggestions for detected diseases.

⚙️ How It Works

The user uploads a plant leaf image through the Streamlit web interface.

The image is preprocessed using OpenCV (resizing, normalization).

A pre-trained CNN model (plant_disease_model.h5) predicts the disease class.

The system displays:

Plant name

Disease name (or healthy status)

Prediction confidence

Disease prevention tips (if applicable)

🧠 Technologies Used

Python

TensorFlow / Keras – Deep Learning model

OpenCV – Image preprocessing

NumPy – Numerical operations

Streamlit – Web application UI

📂 Project Structure
Disease-Detection-main/
│
├── PLANT/
│   ├── main_app.py                # Streamlit application
│   ├── plant_disease_model.h5     # Trained CNN model
│   ├── requirements.txt           # Required dependencies
│   └── Test Images/               # Sample leaf images

🚀 Features

Upload plant leaf images

Real-time disease prediction

Healthy vs Diseased classification

Confidence percentage display

Disease prevention suggestions

User-friendly web interface

🎯 Use Cases

Smart agriculture systems

Farmer decision-support tools

Academic AI/ML projects

Early plant disease diagnosis

▶️ How to Run the Project
pip install -r requirements.txt
streamlit run main_app.py
