import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except Exception:
    RetinaFace = None
    RETINAFACE_AVAILABLE = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False


# ---------------- Configuration ----------------
st.set_page_config(
    page_title="Face & Emotion Detection",
    page_icon="😊",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "models" / "efficientnetb0_emotion_final.keras"
IMAGE_SIZE = 224

# Must match the class order used when the model was trained.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------- Styling ----------------
st.markdown(
    """
    <style>
      .main-title {text-align:center; padding: .35rem 0 .2rem 0;}
      .sub-title {text-align:center; color:#8b949e; margin-bottom:1.4rem;}
      div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.18);
        padding: 12px 14px;
        border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1 class='main-title'>😊 Face & Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>Real-time multi-face detection and seven-class emotion estimation</p>",
    unsafe_allow_html=True,
)


# ---------------- Model loading ----------------
@st.cache_resource(show_spinner="Loading EfficientNetB0 emotion model...")
def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Place the .keras file inside the models folder."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


try:
    emotion_model = load_emotion_model()
except Exception as exc:
    st.error(f"Could not load the emotion model: {exc}")
    st.stop()


# ---------------- Face detection ----------------
@st.cache_resource
def load_haar_detector():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("OpenCV Haar cascade could not be loaded.")
    return cascade


def detect_faces_bgr(frame_bgr, detector="RetinaFace", min_face_size=50):
    """Return face boxes as (x, y, width, height), clipped to the image."""
    image_h, image_w = frame_bgr.shape[:2]
    boxes = []

    if detector == "RetinaFace" and RETINAFACE_AVAILABLE:
        try:
            found = RetinaFace.detect_faces(frame_bgr)
            if isinstance(found, dict):
                for item in found.values():
                    x1, y1, x2, y2 = [int(v) for v in item["facial_area"]]
                    x1 = max(0, min(x1, image_w - 1))
                    y1 = max(0, min(y1, image_h - 1))
                    x2 = max(0, min(x2, image_w))
                    y2 = max(0, min(y2, image_h))
                    w, h = x2 - x1, y2 - y1
                    if w >= min_face_size and h >= min_face_size:
                        boxes.append((x1, y1, w, h))
            return boxes
        except Exception:
            # Keep the application usable if RetinaFace encounters a runtime issue.
            pass

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = load_haar_detector().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


# ---------------- Emotion prediction ----------------
def predict_emotion(face_bgr):
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Empty face crop.")

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)

    # Keep the same EfficientNet preprocessing used by the recovered app.
    batch = tf.keras.applications.efficientnet.preprocess_input(batch)
    probabilities = emotion_model.predict(batch, verbose=0)[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} classes, expected {len(CLASS_NAMES)}."
        )

    index = int(np.argmax(probabilities))
    return CLASS_NAMES[index], float(probabilities[index]), probabilities


def analyze_frame(frame_bgr, detector, min_face_size, confidence_threshold):
    annotated = frame_bgr.copy()
    rows = []
    boxes = detect_faces_bgr(frame_bgr, detector, min_face_size)

    for face_number, (x, y, w, h) in enumerate(boxes, start=1):
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        try:
            emotion, confidence, probabilities = predict_emotion(crop)
            accepted = confidence >= confidence_threshold
            shown_label = emotion if accepted else "uncertain"
            rows.append({
                "Face": face_number,
                "Emotion": emotion if accepted else "Uncertain",
                "Confidence": confidence,
                "Time": datetime.now().strftime("%H:%M:%S"),
                **{name.title(): float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
            })
            color = (40, 200, 80) if accepted else (0, 165, 255)
            label = f"Face {face_number}: {shown_label} ({confidence:.0%})"
        except Exception as exc:
            color = (0, 0, 255)
            label = f"Face {face_number}: prediction error"
            rows.append({
                "Face": face_number, "Emotion": "Error", "Confidence": 0.0,
                "Time": datetime.now().strftime("%H:%M:%S"),
            })

        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            annotated, label, (x, max(24, y-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA,
        )

    return annotated, rows


def rows_to_counts(rows):
    return Counter(row["Emotion"] for row in rows if row["Emotion"] not in ("Error", "Uncertain"))


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Detection settings")
detector_choice = st.sidebar.selectbox(
    "Face detector",
    ["RetinaFace", "Haar Cascade"],
    index=0 if RETINAFACE_AVAILABLE else 1,
)
if detector_choice == "RetinaFace" and not RETINAFACE_AVAILABLE:
    st.sidebar.warning("RetinaFace is unavailable. Haar Cascade will be used.")
    detector_choice = "Haar Cascade"

min_face_size = st.sidebar.slider("Minimum face size", 30, 160, 60, 5)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.40, 0.05)

if not RETINAFACE_AVAILABLE:
    st.sidebar.info("Install the retina-face package to enable RetinaFace detection.")


# ---------------- Tabs ----------------
tab_image, tab_camera, tab_about = st.tabs(["🖼️ Image analysis", "📹 Live camera", "ℹ️ About"])

with tab_image:
    uploaded_file = st.file_uploader(
        "Upload a face image", type=["jpg", "jpeg", "png"], key="image_upload"
    )
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            rgb = np.array(pil_image)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            with st.spinner("Detecting faces and estimating emotions..."):
                annotated, rows = analyze_frame(
                    frame, detector_choice, min_face_size, confidence_threshold
                )
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection result",
                     use_container_width=True)
            if rows:
                counts = rows_to_counts(rows)
                m1, m2 = st.columns(2)
                m1.metric("Faces detected", len(rows))
                m2.metric("Confident predictions", sum(v for v in counts.values()))
                st.subheader("Emotion results")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                if counts:
                    chart_df = pd.DataFrame(
                        [{"Emotion": k.title(), "Faces": v} for k, v in counts.items()]
                    ).set_index("Emotion")
                    st.bar_chart(chart_df)
                csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download image analysis CSV", csv,
                    "image_emotion_results.csv", "text/csv",
                )
            else:
                st.info("No faces detected. Try a clear, front-facing image with good lighting.")
        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")
    else:
        st.info("Upload a JPG or PNG image to analyze one or more faces.")


# ---------------- Live camera processor ----------------
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_rows = []
        self.history = []
        self.frame_index = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.frame_index += 1

        # Analyze every second frame to reduce CPU load.
        if self.frame_index % 2 == 0:
            try:
                annotated, rows = analyze_frame(
                    image, detector_choice, min_face_size, confidence_threshold
                )
                timestamp = datetime.now().strftime("%H:%M:%S")
                with self.lock:
                    self.latest_rows = rows
                    for row in rows:
                        if row["Emotion"] not in ("Error", "Uncertain"):
                            self.history.append({
                                "Time": timestamp,
                                "Face": row["Face"],
                                "Emotion": row["Emotion"],
                                "Confidence": row["Confidence"],
                            })
                    self.history = self.history[-1000:]
                image = annotated
            except Exception:
                pass
        else:
            # Display last camera frame without blocking the stream.
            pass

        return av.VideoFrame.from_ndarray(image, format="bgr24")


with tab_camera:
    st.caption("Allow browser camera access when prompted. The browser sends video to this Streamlit session.")
    ctx = webrtc_streamer(
        key="face-emotion-live",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    if AUTOREFRESH_AVAILABLE and ctx.state.playing:
        st_autorefresh(interval=1500, key="live_dashboard_refresh")

    processor = ctx.video_processor
    if processor is not None:
        with processor.lock:
            live_rows = list(processor.latest_rows)
            live_history = list(processor.history)

        counts = rows_to_counts(live_rows)
        total_faces = len(live_rows)
        dominant = counts.most_common(1)[0][0].title() if counts else "—"

        c1, c2, c3 = st.columns(3)
        c1.metric("Faces in latest analyzed frame", total_faces)
        c2.metric("Dominant emotion", dominant)
        c3.metric("Emotion classes detected", len(counts))

        if counts:
            st.subheader("Live emotion distribution")
            st.bar_chart(pd.DataFrame(
                [{"Emotion": k.title(), "Faces": v} for k, v in counts.items()]
            ).set_index("Emotion"))

        st.subheader("Recent emotion timeline")
        if live_history:
            history_df = pd.DataFrame(live_history)
            st.dataframe(history_df.tail(30), use_container_width=True)
            st.download_button(
                "⬇️ Download live emotion log",
                history_df.to_csv(index=False).encode("utf-8"),
                "live_emotion_log.csv",
                "text/csv",
            )
        else:
            st.info("Start the camera and wait for the first analyzed frame.")
    else:
        st.info("Start the camera to see live emotion statistics and timeline.")


with tab_about:
    st.markdown(
        """
        **Pipeline**
        1. RetinaFace detects face regions (Haar Cascade is available as a fallback).
        2. Each face is resized to 224 × 224 RGB.
        3. The recovered EfficientNetB0 model predicts one of seven emotion classes.
        4. Results are displayed with confidence scores and downloadable CSV logs.

        **Emotion classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

        **Important limitation:** facial-expression models estimate visible expression patterns;
        they do not reliably establish a person's true internal emotional state.
        """
    )

st.caption("Research/demo application. Predictions may vary with lighting, pose, image quality and demographic factors.")
