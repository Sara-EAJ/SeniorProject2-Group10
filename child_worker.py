import os
import cv2
import math
import queue
import joblib
import numpy as np
from ultralytics import YOLO
from core.event_types import create_event, CHILD_ALONE_EVENT

    
# Load both YOLO and ML models 
model = YOLO("models/YOLO-Child.pt") 
ml_model = joblib.load("models/ChildLG.pkl")
    
# Distance threshold used to match detections with previous tracked positions
DIST_THRESHOLD = 180

# ---------------- TIME & SENSITIVITY SETTINGS ----------------
ALONE_THRESHOLD = 3.0       # Time (in seconds) a child must stay alone before triggering alert
ADULT_CONFIRM_TIME = 3.0    # Time (in seconds) adult must be visible to reset alone counter
ADULT_CONF_THRESHOLD = 0.7  # Confidence threshold for adult detection (higher for reliability)
CHILD_CONF_THRESHOLD = 0.3  # Confidence threshold for child detection

TOTAL_ALONE_TIME = 0        # Total accumulated time the child has been alone across the entire video (for ML feature)
MAX_ALONE_TIME = 0          # Longest continuous duration the child stayed alone (captures worst-case scenario)
ADULT_FRAME_COUNT = 0       # Number of frames where at least one reliable adult is detected (used for ratio feature)
TOTAL_FRAMES = 0            # Total number of processed frames in the video (used for normalization)

def explain_child_alone(child_count, adult_count, child_conf, time_alone,threshold):
    explanation = {"reasoning": [] }
    
    # Child detection
    if child_count > 0:
        explanation["reasoning"].append(f"Artificial Intelligence detected {child_count} child in the elevator.")
    
    #Conidence explanation
    explanation["reasoning"].append(f"Artificial Intelligence detection confidence: {round(child_conf,2)}.")

    # Adult detection
    if adult_count == 0:
        explanation["reasoning"].append(f"Adult presence detected: {adult_count}.")

    # Time alone explanation
    explanation["reasoning"].append(f"Child without adult supervision for {round(time_alone,2)} seconds.")

    
    return explanation

# Calculate Euclidean distance between two 2D points.
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

import pandas as pd

def ml_predict(child_count, adult_count, max_alone_time, total_alone_time, adult_presence_ratio):

    X = pd.DataFrame([{
        "num_children": child_count,
        "num_adults": adult_count,
        "max_alone_time": max_alone_time,
        "total_alone_time": total_alone_time,
        "adult_presence_ratio": adult_presence_ratio
    }])

    return ml_model.predict(X)[0]

def child_worker(frame_queue, event_queue, display_queue):
    
    # Flag to control video display
    show = False

    # ---------------- STATE VARIABLES ----------------
    alone_start_time = None         # Timestamp when child started being alone
    adult_appearance_start = None   # Timestamp when dult started appearing
    last_alert_time = 0             # Prevents alert spam (cooldown timer)
    
    
    # ---------------- TRACKING DICTIONARIES ----------------
    id_last_position = {}   # Stores last known position of each unified ID
    byteID_to_unified = {}  # Maps YOLO tracker ID to our custom unified ID
    next_unified_id = 1     # Counter to generate new unified IDs

    while True:
        try:
            # Get frame data from queue
            frame_data = frame_queue.get(timeout=2)
        except queue.Empty: 
            continue
        # Stop worker if None signal received
        if frame_data is None: 
            break

        frame = frame_data["frame"]
        timestamp = frame_data["timestamp"]
        
        # Run YOLO tracking on current frame
        results = model.track(source=frame, 
                            conf=0.3, 
                            persist=True,
                            stream=False,
                            verbose=False
                        )

        current_children_ids = []
        current_adults_ids = []
        max_child_conf = 0

        for r in results:
            if r.boxes is None: continue
            
            # Extract labels, confidence scores, bounding boxes, and tracker IDs
            labels = [model.names[int(c)] for c in r.boxes.cls]
            confs = r.boxes.conf
            boxes = r.boxes.xyxy
            ids = r.boxes.id if r.boxes.id is not None else [None]*len(labels)

            for label, box, conf_score, byte_id in zip(labels, boxes, confs, ids):
                conf_val = float(conf_score)
                
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box)
                
                # Compute center of bounding box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                if byte_id is not None:
                    # If this YOLO ID was never seen before
                    if byte_id not in byteID_to_unified:
                        matched_id = None
                        
                        # Try matching with previous unified IDs
                        for uid, last_pos in id_last_position.items():
                            if euclidean((cx, cy), last_pos) < DIST_THRESHOLD:
                                matched_id = uid
                                break
                            
                        # If no match found, create new unified ID
                        if matched_id is None:
                            matched_id = next_unified_id
                            next_unified_id += 1
                        
                        byteID_to_unified[byte_id] = matched_id
                        
                    # Get unified ID
                    curr_uid = byteID_to_unified[byte_id]
                     # Update last known position
                    id_last_position[curr_uid] = (cx, cy) 
                else:
                    continue  # Ignore detection if no tracking ID

                if label == "child" and conf_val >= CHILD_CONF_THRESHOLD:
                    current_children_ids.append(curr_uid)
                    max_child_conf = max(max_child_conf, conf_val)
                    color = (0, 255, 0)
                elif label == "adult" and conf_val >= ADULT_CONF_THRESHOLD:
                    current_adults_ids.append(curr_uid)
                    color = (0, 0, 255)
                else:
                    continue

                if show:
                    label_str = f"ID:{curr_uid} {label} {conf_val:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label_str, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # ------------- SEND LIVE OUTPUT -------------
                live_output = {
                    "source": "CHILD_WORKER",
                    "timestamp": timestamp,
                    "child_count": len(current_children_ids),
                    "adult_count": len(current_adults_ids),
                    "id": curr_uid,
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf_val, 2)
                }
                display_queue.put(live_output)
                event_queue.put(live_output)
 
        
        has_child = len(current_children_ids) > 0
        has_reliable_adult = len(current_adults_ids) > 0

        # Update frame stats
        global TOTAL_FRAMES, ADULT_FRAME_COUNT

        TOTAL_FRAMES += 1

        if has_reliable_adult:
            ADULT_FRAME_COUNT += 1
        if has_child:
            if has_reliable_adult:
                
                # Start counting adult presence duration
                if adult_appearance_start is None:
                    adult_appearance_start = timestamp
                
                # If adult confirmed for enough time, reset alone counter
                if (timestamp - adult_appearance_start) >= ADULT_CONFIRM_TIME:
                    alone_start_time = None 
                global TOTAL_ALONE_TIME
                TOTAL_ALONE_TIME += (timestamp - alone_start_time)
                
            else:
                # No reliable adult detected
                adult_appearance_start = None 
                if alone_start_time is None:
                    alone_start_time = timestamp
                
                elapsed_alone = timestamp - alone_start_time
                global MAX_ALONE_TIME
                MAX_ALONE_TIME = max(MAX_ALONE_TIME, elapsed_alone)
                
                # Trigger alert if child alone for too long
                if elapsed_alone >= ALONE_THRESHOLD:
                    adult_presence_ratio = ADULT_FRAME_COUNT / max(TOTAL_FRAMES, 1)
                    ml_decision = ml_predict(
                        child_count=len(current_children_ids),
                        adult_count=len(current_adults_ids),
                        max_alone_time=MAX_ALONE_TIME,
                        total_alone_time=TOTAL_ALONE_TIME,
                        adult_presence_ratio=adult_presence_ratio
                    )

                    if ml_decision == 1:
                        xai_data = explain_child_alone(
                        child_count=len(current_children_ids),
                        adult_count=len(current_adults_ids),
                        child_conf=max_child_conf,
                        time_alone=elapsed_alone,
                        threshold=ALONE_THRESHOLD
                    )
                    event = create_event(CHILD_ALONE_EVENT, timestamp, max_child_conf)
                    event["xai"] = xai_data
                    event_queue.put(event)
                    
                    last_alert_time = timestamp
                    
        else:
            alone_start_time = None
            adult_appearance_start = None

        if show:
            cv2.imshow("Secure Child Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    event_queue.put(None)
    cv2.destroyAllWindows()
    print("Child worker stopped.")