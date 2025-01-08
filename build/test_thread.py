# test_thread.py
import sys
import os

# Add the project directory to sys.path
dir = r"C:\Users\Azd_A\OneDrive\Desktop\TrafficLightManagementSystem"
sys.path.append(dir)

import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import traci

from rl_helpers import (
    get_all_red_phase_index,
    get_enriched_state,
    compute_reward,
    update_shared_metrics,
    POSSIBLE_DURATIONS,
    NUM_DURATIONS,
    MIN_GREEN_TIME,
    ALL_RED_TIME,
    VEHICLE_THRESHOLD
)
from shared_data import shared_emergency, shared_emergency_lock
from models import EnhancedDQNAgent
from replay_buffer import PrioritizedReplayBuffer  # Import the class

BEST_MODEL_PATH = r"C:\Users\Azd_A\Downloads\prioritized_no_repicking_best_new.pth"

def test_script_thread():
    """
    Thread function for test.py-like logic.
    We do NOT start or close SUMO here; it's started once in main().
    We just retrieve the 'shared_conn' connection.
    """
    # Configure logging: file only
    logging.basicConfig(
        filename='my_log_file.log',
        filemode='w',  # overwrite each run
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    logger = logging.getLogger(__name__)

    # Use the existing connection
    conn = traci.getConnection("shared_conn")

    # Grab Traffic Light (TL) IDs
    tl_ids = list(conn.trafficlight.getIDList())
    if not tl_ids:
        raise ValueError("No traffic lights found in the network!")

    n_tl = len(tl_ids)
    n_phases = len(conn.trafficlight.getAllProgramLogics(tl_ids[0])[0].phases)

    all_red_indices = {}
    for tid in tl_ids:
        aidx = get_all_red_phase_index(tid, conn)
        if aidx is None:
            raise ValueError(f"No all-red phase for TL {tid}!")
        all_red_indices[tid] = aidx

    # Determine state dimension
    state_size = 0
    for tid in tl_ids:
        lanes = conn.trafficlight.getControlledLanes(tid)
        # each lane => (normalized queue, normalized count, is_stuck)
        state_size += len(lanes) * 3

    # Load the RL Agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize the agent
    agent = EnhancedDQNAgent(state_size, n_tl, n_phases)

    # Load the checkpoint with proper device mapping
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the checkpoint was saved differently, adjust accordingly
        agent.load_state_dict(checkpoint)

    # Move the agent to the device
    agent.to(device)
    agent.eval()

    # Verify device assignment
    print(f"Agent is on device: {next(agent.parameters()).device}")

    epsilon = 0.0
    gamma = 0.99

    # Force all TLs to all-red for ALL_RED_TIME initially
    for tl_id in tl_ids:
        conn.trafficlight.setPhase(tl_id, all_red_indices[tl_id])
        conn.trafficlight.setPhaseDuration(tl_id, ALL_RED_TIME)
    for _ in range(int(ALL_RED_TIME)):
        conn.simulationStep()

    current_green_tl = None
    last_switch_time = 0.0
    step = 0
    max_steps = 20000
    step_rewards = []
    step_losses = []
    total_reward = 0.0

    state = get_enriched_state(tl_ids, conn).to(device)

    while step < max_steps:
        curr_time = conn.simulation.getTime()

        # -------------------------
        # 1) Check for emergency vehicles => override
        # -------------------------
        emergency_vehicles = [
            v for v in conn.vehicle.getIDList()
            if conn.vehicle.getTypeID(v) == "emergency"
        ]
        if emergency_vehicles:
            emerg_veh_id = emergency_vehicles[0]
            emerg_lane_id = conn.vehicle.getLaneID(emerg_veh_id)
            controlling_tl_id = None
            for tid in tl_ids:
                lanes = conn.trafficlight.getControlledLanes(tid)
                if emerg_lane_id in lanes:
                    controlling_tl_id = tid
                    break

            logger.info(
                f"EMERGENCY OVERRIDE: Vehicle '{emerg_veh_id}' in lane '{emerg_lane_id}' => "
                f"TL '{controlling_tl_id}' forced green."
            )
            # Update shared metrics (important!)
            shared_emergency_lock.acquire()
            shared_emergency["emerg_veh_id"] = emerg_veh_id
            shared_emergency["emerg_lane_id"] = emerg_lane_id
            shared_emergency["controlling_tl_id"] = controlling_tl_id
            shared_emergency_lock.release()

            if controlling_tl_id is not None:
                # Force controlling TL green, others red
                for tid in tl_ids:
                    if tid == controlling_tl_id:
                        conn.trafficlight.setPhase(tid, 0)  # assume 0 => green
                        conn.trafficlight.setPhaseDuration(tid, 9999)
                    else:
                        conn.trafficlight.setPhase(tid, all_red_indices[tid])
                        conn.trafficlight.setPhaseDuration(tid, 9999)

                conn.simulationStep()
                step += 1

                r_emerg = compute_reward(tl_ids, conn)
                total_reward += r_emerg
                step_rewards.append(r_emerg)

                # Update shared metrics (important!)
                update_shared_metrics(conn, step)

                if conn.simulation.getMinExpectedNumber() == 0:
                    break
                continue

        # -------------------------
        # 2) Normal RL flow
        # -------------------------
        time_in_current_green = (curr_time - last_switch_time) if current_green_tl else 0.0
        can_switch = (time_in_current_green >= MIN_GREEN_TIME)

        with torch.no_grad():
            q_values = agent(state.unsqueeze(0)).squeeze(0)

        # Epsilon-greedy
        if random.random() < epsilon:
            action_idx = random.randint(0, q_values.numel() - 1)
        else:
            action_idx = torch.argmax(q_values).item()

        num_actions_per_tl = n_phases * NUM_DURATIONS
        chosen_tl_idx = action_idx // num_actions_per_tl
        remainder = action_idx % num_actions_per_tl
        chosen_phase_idx = remainder // NUM_DURATIONS
        duration_idx = remainder % NUM_DURATIONS
        chosen_duration = POSSIBLE_DURATIONS[duration_idx]
        new_tl_id = tl_ids[chosen_tl_idx]

        if current_green_tl is None:
            # No green => set new TL green
            for tid in tl_ids:
                if tid == new_tl_id:
                    conn.trafficlight.setPhase(tid, chosen_phase_idx)
                    conn.trafficlight.setPhaseDuration(tid, chosen_duration)
                else:
                    conn.trafficlight.setPhase(tid, all_red_indices[tid])
                    conn.trafficlight.setPhaseDuration(tid, 9999)
            current_green_tl = new_tl_id
            last_switch_time = curr_time
        else:
            # Switch if it's a different TL and we can switch
            if new_tl_id != current_green_tl and can_switch:
                for tid in tl_ids:
                    if tid == new_tl_id:
                        conn.trafficlight.setPhase(tid, chosen_phase_idx)
                        conn.trafficlight.setPhaseDuration(tid, chosen_duration)
                    else:
                        conn.trafficlight.setPhase(tid, all_red_indices[tid])
                        conn.trafficlight.setPhaseDuration(tid, 9999)
                current_green_tl = new_tl_id
                last_switch_time = curr_time

        conn.simulationStep()
        step += 1

        # -------------------------
        # 3) Compute reward
        # -------------------------
        r = compute_reward(tl_ids, conn)
        total_reward += r
        step_rewards.append(r)

        # -------------------------
        # 4) Simple TD error (just logs)
        # -------------------------
        current_q_of_action = q_values[action_idx].item()
        next_state = get_enriched_state(tl_ids, conn).to(device)
        with torch.no_grad():
            next_q = agent(next_state.unsqueeze(0)).squeeze(0)
        max_next_q = next_q.max().item()

        td_error = r + gamma * max_next_q - current_q_of_action
        td_error = torch.tensor(td_error, device=device)  # Move TD error to GPU
        step_losses.append(abs(td_error.cpu().item()))  # Move back to CPU for logging

        state = next_state

        # -------------------------
        # 5) "A lot of vehicles" log
        # -------------------------
        for tid in tl_ids:
            lanes = conn.trafficlight.getControlledLanes(tid)
            total_count_for_tl = sum(conn.lane.getLastStepVehicleNumber(lane_id) for lane_id in lanes)
            if total_count_for_tl > VEHICLE_THRESHOLD:
                logger.info(
                    f"A lot of vehicles at TL '{tid}': total={total_count_for_tl} lanes={lanes}."
                )

        # -------------------------
        # Update shared metrics for FastAPI
        # -------------------------
        update_shared_metrics(conn, step)

        if conn.simulation.getMinExpectedNumber() == 0:
            break

    final_time = conn.simulation.getTime()
    logger.info(f"Done. Steps={step}, Final time={final_time:.2f}, Total reward={total_reward:.2f}")
