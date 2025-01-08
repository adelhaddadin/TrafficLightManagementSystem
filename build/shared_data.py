# shared_data.py

import threading

##############################################################################
# Shared global dictionaries and locks
##############################################################################
shared_metrics = {
    "time": 0.0,
    "co2": 0.0,
    "nox": 0.0,
    "pmx": 0.0,
    "vehicle_count": 0,
    "average_speed": 0.0,
    "status": "running"
}
shared_metrics_lock = threading.Lock()

shared_inference = {
    "A": 0.0,
    "B": 0.0,
    "C": 0.0,
    "D": 0.0
}
shared_inference_lock = threading.Lock()

shared_emergency = {
    "emerg_veh_id": 0,
    "emerg_lane_id": 0,
    "controlling_tl_id": 0
}
shared_emergency_lock = threading.Lock()

