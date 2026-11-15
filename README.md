# Face & Emotion Detection using RetinaFace and EfficientNetB0

## Project files
- `app.py`: Streamlit application for image upload and live webcam analysis.
- `models/efficientnetb0_emotion_final.keras`: recovered trained emotion model.
- `requirements.txt`: Python dependencies.

## Setup (Windows 10/11)
Use Python 3.10 or 3.11 in a fresh virtual environment.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```powershell
streamlit run app.py
```

Open the local URL shown in the terminal. Allow camera access for live mode.

## Notes
- The model's class order is configured as: angry, disgust, fear, happy, neutral, sad, surprise.
- The app preserves the preprocessing call used by the recovered application.
- RetinaFace is preferred; Haar Cascade is available as a fallback.
- Webcam access is provided by `streamlit-webrtc`; localhost is the easiest place to test it.
- If TensorFlow cannot load the `.keras` model, record the full error and Python/TensorFlow versions before changing or re-saving the model.
- This is an expression-estimation demo, not a reliable measure of a person's internal emotional state.
