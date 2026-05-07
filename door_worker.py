import cv2
import numpy as np
import collections
import queue
import time
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from core.event_types import create_event, DOOR_FAULT_EVENT


def analyze_door_motion(states, motion, motion_threshold=0.3):

    transitions = 0
    cycles = 0

    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            transitions += 1
            if states[i - 1] == 0 and states[i] == 1:
                cycles += 1

    avg_motion = np.mean(motion) if len(motion) > 0 else 0

    stuck_open = states.count(1) > len(states) * 0.6 and avg_motion < motion_threshold
    stuck_semi = states.count(2) > len(states) * 0.5
    lagging = transitions > 8

    status = "FAULTY" if (stuck_open or stuck_semi or lagging) else "NORMAL"

    return {
        "status": status,
        "cycles": cycles,
        "transitions": transitions,
        "avg_motion": avg_motion,
        "stuck_open": stuck_open,
        "stuck_semi": stuck_semi,
        "lagging": lagging
    }


def explain_door_fault(analysis):
    reasons = []

    if analysis["stuck_open"]:
        reasons.append("Door remained open without movement")

    if analysis["stuck_semi"]:
        reasons.append("Door remained semi open without movement")

    if analysis["lagging"]:
        reasons.append("Door state changing excessively")

    if not reasons:
        return "Door behavior normal"

    return " | ".join(reasons)


WINDOW_SIZE = 20
CONF_THRES = 0.5
ALERT_THRESHOLD = 3


def door_worker(frame_queue, event_queue, display_queue):

    print("Door Worker Started")

    # YOLO
    yolo_model = YOLO("models/YOLO-Door.pt")
    CLASS_NAMES = ["close_door", "open_door", "semi_door"]
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 165, 255)]

    # BiLSTM
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(WINDOW_SIZE, 6)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.load_weights("models/DoorBiLSTM.h5")

    # Buffers
    sequence = []
    last_valid = None
    lag_counter = 0

    prev_gray = None
    fps = None
    FRAMES_TO_HOLD = None

    state_buffer = None
    motion_buffer = None

    stable_state = 0
    start_time = time.time()
    frame_counter = 0

    while True:
        try:
            frame_data = frame_queue.get(timeout=2)
        except queue.Empty:
            continue

        if frame_data is None:
            break

        frame = frame_data["frame"]
        timestamp = frame_data["timestamp"]

        frame_counter += 1

        # ===== FPS ESTIMATION =====
        if fps is None and frame_counter >= 20:
            elapsed = time.time() - start_time
            fps = frame_counter / elapsed if elapsed > 0 else 25
            FRAMES_TO_HOLD = max(1, int(5 * fps))

            state_buffer = collections.deque(maxlen=FRAMES_TO_HOLD)
            motion_buffer = collections.deque(maxlen=FRAMES_TO_HOLD)

        # ===== YOLO =====
        results = yolo_model(frame, imgsz=640, verbose=False)[0]

        raw_state = 0
        best_conf = 0.0

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf
                raw_state = cls
                best = {
                    "bbox": box.xyxy[0].cpu().numpy(),
                    "class": cls,
                    "confidence": conf
                }

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[cls], 2)
            cv2.putText(frame, f"{CLASS_NAMES[cls]} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)

        # ===== STATE BUFFER =====
        if state_buffer is not None:
            state_buffer.append(raw_state)

            if len(state_buffer) == FRAMES_TO_HOLD:
                counts = {0: 0, 1: 0, 2: 0}
                for s in state_buffer:
                    counts[s] += 1
                stable_state = max(counts, key=counts.get)

        current_state = stable_state if state_buffer else raw_state

        # ===== OPTICAL FLOW =====
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and motion_buffer is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_buffer.append(np.mean(mag))

        prev_gray = gray

        # ===== ANALYSIS =====
        optical_flow_lag = False
        reason = ""
        avg_motion = 0

        if state_buffer is not None and len(state_buffer) == FRAMES_TO_HOLD:

            analysis = analyze_door_motion(
                list(state_buffer),
                list(motion_buffer),
                0.3
            )

            optical_flow_lag = analysis["status"] == "FAULTY"
            reason = explain_door_fault(analysis)
            avg_motion = analysis["avg_motion"]

        # ===== BiLSTM =====
        if 'best' in locals():
            feature = list(best["bbox"]) + [best["class"], best["confidence"]]
            last_valid = feature
        else:
            feature = last_valid if last_valid else [0, 0, 0, 0, -1, 0]

        sequence.append(feature)

        if len(sequence) < WINDOW_SIZE:
            display_queue.put({"frame": frame})
            continue

        if len(sequence) > WINDOW_SIZE:
            sequence.pop(0)

        input_seq = np.array(sequence).reshape(1, WINDOW_SIZE, 6)
        pred = model.predict(input_seq, verbose=0)

        model_class = int(np.argmax(pred))
        model_conf = float(np.max(pred))

        if model_class == 1:
            lag_counter += 1
        else:
            lag_counter = 0

        model_lag = lag_counter >= ALERT_THRESHOLD

        # ===== STRICT AND =====
        if optical_flow_lag and model_lag:
            status = "FAULTY"
        else:
            status = "NORMAL"

        # ===== OUTPUT =====
        door_msg = {
            "worker": "door",
            "status": status,
            "timestamp": timestamp,
            "confidence": model_conf,
            "optical_flow_lag": optical_flow_lag,
            "model_lag": model_lag,
            "reason": reason
        }

        event_queue.put(door_msg)
        display_queue.put(door_msg)

        # ===== ALERT =====
        if status == "FAULTY":
            alert_frame = frame.copy()

            display_queue.put({
                "worker": "door_alert",
                "frame": alert_frame,
                "status": status
            })

            event = create_event(DOOR_FAULT_EVENT, timestamp, model_conf)
            event["xai"] = {
                "reasoning": [
                    "Optical Flow AND BiLSTM both detected fault",
                    reason
                ]
            }

            event_queue.put(event)
            display_queue.put(event)

        # ===== DISPLAY =====
        cv2.putText(frame, f"System: {status}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255), 2)

        cv2.putText(frame, f"Flow:{optical_flow_lag} Model:{model_lag}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0), 2)

        display_queue.put({"frame": frame})

    event_queue.put(None)
    cv2.destroyAllWindows()
    print("Worker stopped")