import time
import winsound

last_alert_time = {}

def check_alert(label, cooldown=10):
    now = time.time()
    if label not in last_alert_time or (now - last_alert_time[label]) > cooldown:
        last_alert_time[label] = now
        return True
    return False

def play_alert_sound():
    winsound.Beep(1000, 500)

def classify_event(detections):
    labels = [d["label"] for d in detections]

    if "knife" in labels:
        return "CRIME DETECTED: Weapon Spotted"
    elif len([l for l in labels if l == "person"]) >= 4:
        return "CROWD INCIDENT: Multiple Persons"
    elif "car" in labels or "truck" in labels or "bus" in labels:
        return "VEHICLE DETECTED: Possible Accident Zone"
    else:
        return None