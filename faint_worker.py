from pyexpat import model

import cv2
import time
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from core.event_types import create_event, FAINT_EVENT

WINDOW_SIZE = 15
FRAME_SKIP = 2
CONF_THRESHOLD = 0.80

def faint_worker(frame_queue, event_queue, display_queue):
    print("Faint worker started")

    # LSTM model
    model = Sequential([
        Conv1D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
            input_shape=(WINDOW_SIZE, 66)
        ),

        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same'
        ),

        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Bidirectional(
            LSTM(
                64,
                return_sequences=True
            )
        ),

        Dropout(0.5),

        Bidirectional(
            LSTM(
                32,
                return_sequences=False
            )
        ),

        Dropout(0.5),

        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.load_weights("models/CNNBiLSTM-Faint.h5")

    print("Faint model loaded successfully")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3 )

    keypoints_buffer = []
    frame_count = 0
    last_event_time = 0
    COOLDOWN = 5
    last_valid_keypoints = None

    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            break

        frame = frame_data["frame"]
        timestamp = frame_data["timestamp"]

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)


        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append(lm.x)
                keypoints.append(lm.y)

            last_valid_keypoints = keypoints
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        else:
            landmarks = None
            if last_valid_keypoints is not None:
                keypoints = last_valid_keypoints
            else:
                keypoints = np.zeros(66)

        keypoints_buffer.append(keypoints)
        if len(keypoints_buffer) > WINDOW_SIZE:
            keypoints_buffer = keypoints_buffer[-WINDOW_SIZE:]

        if len(keypoints_buffer) == WINDOW_SIZE:
            sequence = np.array(keypoints_buffer).reshape(1, WINDOW_SIZE, 66)
            prediction = model.predict(sequence, verbose=0)
            label_idx = np.argmax(prediction)
            confidence = float(np.max(prediction))
            current_prediction = "FAINT" if label_idx == 0 else "NORMAL"

            live_output = {
                "source": "FAINT_WORKER",
                "timestamp": timestamp,
                "label": current_prediction,
                "confidence": confidence,
                "landmarks": landmarks,
                "frame": frame
            }

            event_queue.put(live_output)
            display_queue.put(live_output)

            if label_idx == 0 and confidence > CONF_THRESHOLD:
                if time.time() - last_event_time > COOLDOWN:

                    xai_data = {
                        "reasoning": [
                            "Artificial Intelligence system detected a sudden body collapse posture.",
                            f"Artificial Intelligence detection confidence: {round(confidence,2)}.",
                            "Observed posture pattern is consistent with a fainting event.",
                        ]
                    }

                    event = create_event(
                        event_type=FAINT_EVENT,
                        timestamp=timestamp,
                        confidence=confidence
                    )

                    event["xai"] = xai_data

                    event_queue.put(event)

                    last_event_time = time.time()

                    print("FAINT EVENT SENT:", event)

    pose.close()
    event_queue.put(None)
    cv2.destroyAllWindows()
    print("Faint worker stopped")