😊 Face Detection and Emotion Recognition Using Machine Learning

A real-time facial emotion detection application built using Python, TensorFlow, EfficientNetB0, OpenCV, and Streamlit.

📌 Project Overview

This project detects human faces and estimates facial expressions using deep learning techniques. It provides an interactive web application for image-based emotion analysis and real-time webcam detection.

The system uses EfficientNetB0 for emotion classification and RetinaFace or Haar Cascade for face detection.

✨ Features

Multi-face detection

Seven-class emotion classification

Image upload and emotion analysis

Real-time webcam detection

Emotion confidence scores

Emotion distribution visualization

Timestamp-based emotion tracking

CSV report export

Interactive Streamlit interface

🧠 Emotion Classes

The model classifies facial expressions into seven categories:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

🛠️ Technologies Used

Python

TensorFlow & Keras

EfficientNetB0

OpenCV

RetinaFace

Haar Cascade

NumPy & Pandas

Streamlit

Streamlit-WebRTC

📂 Project Structure

Face-Emotion-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── efficientnetb0_emotion_final.keras
│
└── assets/
    ├── image-analysis.png
    └── live-camera.png

⚙️ Installation and Setup

1. Clone the repository

git clone YOUR_REPOSITORY_URL

2. Navigate to the project directory

cd Face-Emotion-Detection

3. Create a virtual environment

python -m venv .venv

4. Activate the environment (Windows)

.venv\Scripts\activate

5. Install dependencies

pip install -r requirements.txt

6. Run the application

python -m streamlit run app.py

The application will open in your default web browser.


🎯 Project Objectives

Implement deep learning for facial emotion classification.

Integrate face detection with emotion recognition.

Develop a real-time computer vision application.

Provide an interactive dashboard for emotion analysis.

⚠️ Limitations

Emotion predictions are estimates based on visible facial expression patterns and do not necessarily represent a person's actual emotional state. Performance may vary depending on lighting, facial orientation, and image quality.

🚀 Future Improvements

Improve emotion classification accuracy.

Optimize real-time detection performance.

Enhance face tracking and emotion history.

Explore additional deep learning architectures.

👨‍💻 Author

Abhirup Ghosh

BCA Student

📄 License

This project is intended for educational and research purposes. Model and dataset redistribution must follow their respective licenses and permissions.
