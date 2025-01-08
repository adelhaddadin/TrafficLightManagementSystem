# rl_helpers.py
import sys

dir = r"C:/Users/Azd_A/OneDrive/Desktop/TrafficLightManagementSystem"
sys.path.append(dir)

import numpy as np
import torch
import matplotlib
from torch import device
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared_data import shared_metrics, shared_metrics_lock, shared_emergency, shared_emergency_lock

MIN_GREEN_TIME = 5
YELLOW_TIME = 5
ALL_RED_TIME = 5
VEHICLE_THRESHOLD = 7
POSSIBLE_DURATIONS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
NUM_DURATIONS = len(POSSIBLE_DURATIONS)
total_co2 = 0.0
total_nox = 0.0
total_pmx = 0.0
total_speed = 0.0


def get_all_red_phase_index(tl_id, conn):
    """Retrieve the index of an all-red phase from a traffic light's program."""
    phases = conn.trafficlight.getAllProgramLogics(tl_id)[0].phases
    for i, p in enumerate(phases):
        if all(signal == 'r' for signal in p.state):
            return i
    return None

def get_enriched_state(tl_ids, conn) -> torch.Tensor:
    """
    Build a state vector (example):
    - queue length per lane, normalized
    - vehicle count per lane, normalized
    - is_stuck (1 if queue > 10)
    """
    max_queue_global = 1e-8
    max_count_global = 1e-8

    for tl_id in tl_ids:
        lanes = conn.trafficlight.getControlledLanes(tl_id)
        local_queues = [conn.lane.getLastStepHaltingNumber(l) for l in lanes]
        local_counts = [conn.lane.getLastStepVehicleNumber(l) for l in lanes]
        if local_queues:
            max_queue_global = max(max_queue_global, max(local_queues))
        if local_counts:
            max_count_global = max(max_count_global, max(local_counts))

    state = []
    for tl_id in tl_ids:
        lanes = conn.trafficlight.getControlledLanes(tl_id)
        queue_lengths = [conn.lane.getLastStepHaltingNumber(l) for l in lanes]
        vehicle_counts = [conn.lane.getLastStepVehicleNumber(l) for l in lanes]

        normalized_queues = [
            q / max_queue_global if max_queue_global > 0 else 0
            for q in queue_lengths
        ]
        normalized_counts = [
            c / max_count_global if max_count_global > 0 else 0
            for c in vehicle_counts
        ]
        is_stuck_list = [1 if q > 10 else 0 for q in queue_lengths]

        state.extend(normalized_queues)
        state.extend(normalized_counts)
        state.extend(is_stuck_list)

    return torch.tensor(state, dtype=torch.float32)

def compute_reward(traffic_light_ids, conn, fairness_weight=1.0) -> float:
    """
    Mirrors the reward function:
      - For each TL: intersection_penalty = -waiting_time - 2*queue_length
      - Then subtract fairness_weight * np.std(rewards) if multiple TLs
    """
    rewards = []
    for tl_id in traffic_light_ids:
        lanes = conn.trafficlight.getControlledLanes(tl_id)
        waiting_time = sum(conn.lane.getWaitingTime(lane) for lane in lanes)
        queue_length = sum(conn.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        intersection_penalty = -waiting_time - 2 * queue_length
        rewards.append(intersection_penalty)

    rewards = torch.tensor(rewards, device=device)

    # Perform GPU-based operations
    fairness_penalty = fairness_weight * rewards.std() if len(rewards) > 1 else 0.0
    total_reward = rewards.sum() - fairness_penalty

    return total_reward

def compute_reward(traffic_light_ids, conn, fairness_weight=1.0, device='cpu') -> torch.Tensor:
    rewards = []
    for tl_id in traffic_light_ids:
        lanes = conn.trafficlight.getControlledLanes(tl_id)
        waiting_time = sum(conn.lane.getWaitingTime(lane) for lane in lanes)
        queue_length = sum(conn.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        intersection_penalty = -waiting_time - 2 * queue_length
        rewards.append(intersection_penalty)

    rewards = torch.tensor(rewards, device=device)
    fairness_penalty = fairness_weight * rewards.std() if len(rewards) > 1 else 0.0
    total_reward = rewards.sum() - fairness_penalty
    return total_reward

def update_shared_metrics(conn, step):
    """
    Gather aggregate emission data and store in shared_metrics.
    We use a lock to avoid concurrency issues if FastAPI reads at same time.
    """
    with shared_metrics_lock:
        vehicle_ids = conn.vehicle.getIDList()
        vehicle_count = len(vehicle_ids)
        total_co2 = shared_metrics["co2"]
        total_nox = shared_metrics["nox"]
        total_pmx = shared_metrics["pmx"]
        total_speed = shared_metrics["average_speed"]

        for v_id in vehicle_ids:
            total_co2 += conn.vehicle.getCO2Emission(v_id)
            total_nox += conn.vehicle.getNOxEmission(v_id)
            total_pmx += conn.vehicle.getPMxEmission(v_id)
            total_speed += conn.vehicle.getSpeed(v_id)

        avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0.0

        shared_metrics["time"] = float(conn.simulation.getTime())
        shared_metrics["co2"] = total_co2
        shared_metrics["nox"] = total_nox
        shared_metrics["pmx"] = total_pmx
        shared_metrics["vehicle_count"] = vehicle_count
        shared_metrics["average_speed"] = avg_speed
