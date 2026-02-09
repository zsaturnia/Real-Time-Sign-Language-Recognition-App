import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
load_model = tf.keras.models.load_model

st.set_page_config(page_title="ASL Recognition", layout='wide')

@st.cache_resource
def load_keras_model():
    """Load the Keras model from disk, cached for performance"""
    try:
        return load_model('sign_language_model.h5')
    except Exception as e:
        st.error(f"Error loading model : {e}")
        return None
    
model= load_keras_model()

mp_hands = mp.solutions.hands
hands= mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

st.title("Real-Time ASL Letter Recognition")
st.write("Point your hand at the camera to see the mùagic happen !")

run = st.checkbox('Start webcam', value=True)
FRAM_WINDOW = st.image([])

if model is None:
    st.warning("Model not loaded. Please ensure 'sign_language_model.h5' is the directory.")
else:
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture image form camera.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        prediction_text = "No hand detected"

        if results.multi_hand_landmarks :
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                padding = 0.05
                x_min_px = int((x_min - padding) * w)
                x_max_px = int((x_max + padding) * w)
                y_min-px = int((y_min - padding) * w)
                y_max_px = int((y_max + padding) * w)

                