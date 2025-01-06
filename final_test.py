import os
import random
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import traci
import sumolib

# Local imports
from agent import EnhancedDQNAgent
from simulation import (
    POSSIBLE_DURATIONS,
    NUM_DURATIONS
)
from replay_memory import PrioritizedReplayBuffer, SumTree

# ------------------------------
# Constants & Paths
# ------------------------------
SUMO_CONFIG_PATH = r"C:\Program Files (x86)\Eclipse\Sumo\bin\simulation_test\city_network.sumocfg"
BEST_MODEL_PATH = r"C:\Users\adelh\PycharmProjects\pythonProject2\prioritized_no_repicking_best_new.pth"

# We enforce:
# - Only one traffic light is GREEN at a time
# - Min GREEN time = 5 sec
# - Epsilon = 0 (pure greedy)
# - Strict Yellow Phase = 5s
# - Strict All-Red Phase = 5s (for transitions)
MIN_GREEN_TIME = 5
YELLOW_TIME = 5
ALL_RED_TIME = 5
START_TIME = 7500
END_TIME = 12000
# Logging
logging.basicConfig(
    filename='my_log_file.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


def get_all_red_phase_index(tl_id, conn):
    """
    Retrieve the index of an all-red phase from a traffic light's program.
    We assume each TL has exactly one "all red" state.
    """
    phases = conn.trafficlight.getAllProgramLogics(tl_id)[0].phases
    for i, p in enumerate(phases):
        if all(signal == 'r' for signal in p.state):
            return i
    return None


def get_enriched_state(tl_ids, conn) -> torch.Tensor:
    """
    Build a state vector, matching the "get_enriched_state" logic from simulation.py:
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
    rewards = []
    for tl_id in traffic_light_ids:
        lanes = conn.trafficlight.getControlledLanes(tl_id)
        waiting_time = sum(conn.lane.getWaitingTime(lane) for lane in lanes)
        queue_length = sum(conn.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        intersection_penalty = -waiting_time - 2 * queue_length
        rewards.append(intersection_penalty)

    if len(rewards) > 1:
        fairness_penalty = fairness_weight * np.std(rewards)
    else:
        fairness_penalty = 0.0
    total_reward = sum(rewards) - fairness_penalty
    return total_reward


def run_drl_simulation():
    """
    Test script with overriding logic for emergency vehicles if any are in the network
    """
    sumoBinary = sumolib.checkBinary('sumo-gui')
    sumo_cmd = [
        sumoBinary,
        "-c", SUMO_CONFIG_PATH,
        "--start",
        "--quit-on-end",
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--verbose", "true",
        "--begin", str(START_TIME),
        "--end", str(END_TIME)
    ]

    # ============ Initialization Run ============
    traci.start(sumo_cmd, label="init_test")
    conn_init = traci.getConnection("init_test")
    cumulative_co2 = 0.0
    cumulative_nox = 0.0
    cumulative_pm10 = 0.0
    total_waiting_time = 0.0
    drl_waiting_time = []
    drl_vehicle_count = []
    drl_steps = []
    time_series_waiting_time = []
    time_series_vehicle_count = []
    time_series_steps = []
    tl_ids = list(conn_init.trafficlight.getIDList())
    if not tl_ids:
        raise ValueError("No traffic lights found in the network!")

    n_tl = len(tl_ids)
    n_phases = len(conn_init.trafficlight.getAllProgramLogics(tl_ids[0])[0].phases)

    all_red_indices = {}
    for tid in tl_ids:
        ridx = get_all_red_phase_index(tid, conn_init)
        if ridx is None:
            raise ValueError(f"No all-red phase found for TL {tid}!")
        all_red_indices[tid] = ridx

    state_size = 0
    for tid in tl_ids:
        lanes = conn_init.trafficlight.getControlledLanes(tid)
        state_size += len(lanes) * 3

    conn_init.close()

    # ============ Load Pretrained Agent ============
    agent = EnhancedDQNAgent(state_size, n_tl, n_phases)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=torch.device('cpu'))
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()

    epsilon = 0.0
    gamma = 0.99  # used for TD error measure
    # ============ Start the Actual Test Run ============
    traci.start(sumo_cmd, label="test_run")
    conn = traci.getConnection("test_run")
    for tl_id in tl_ids:
        conn.trafficlight.setPhase(tl_id, all_red_indices[tl_id])
        conn.trafficlight.setPhaseDuration(tl_id, ALL_RED_TIME)
    for _ in range(ALL_RED_TIME):
        conn.simulationStep()

    green_start_time = {tid: None for tid in tl_ids}
    total_green_time = {tid: 0.0 for tid in tl_ids}

    current_green_tl = None
    last_switch_time = 0.0

    step = 0
    max_steps = 4500

    step_rewards = []
    step_losses = []
    total_reward = 0.0

    state = get_enriched_state(tl_ids, conn)

    while step < max_steps:
        curr_time = conn.simulation.getTime()
        veh_ids = conn.vehicle.getIDList()
        cumulative_co2 += sum(conn.vehicle.getCO2Emission(vid) for vid in veh_ids)
        cumulative_nox += sum(conn.vehicle.getNOxEmission(vid) for vid in veh_ids)
        cumulative_pm10 += sum(conn.vehicle.getPMxEmission(vid) for vid in veh_ids)
        veh_ids = conn.vehicle.getIDList()
        total_waiting_time = sum(conn.vehicle.getWaitingTime(vid) for vid in veh_ids)
        vehicle_count = len(veh_ids)

        drl_waiting_time.append(total_waiting_time)
        drl_vehicle_count.append(vehicle_count)
        drl_steps.append(curr_time)

        # ------------- EMERGENCY OVERRIDE LOGIC -------------
        # 1) Check if any emergency vehicles exist
        emergency_vehicles = [
            v for v in conn.vehicle.getIDList()
            if conn.vehicle.getTypeID(v) == "emergency"
        ]

        if emergency_vehicles:
            emerg_veh_id = emergency_vehicles[0]
            emerg_lane_id = conn.vehicle.getLaneID(emerg_veh_id)
            #find which TL controls that lane:
            controlling_tl_id = None
            for tid in tl_ids:
                lanes = conn.trafficlight.getControlledLanes(tid)
                if emerg_lane_id in lanes:
                    controlling_tl_id = tid
                    break

            logger.info(
                f"Detected emergency vehicle '{emerg_veh_id}' in lane '{emerg_lane_id}'. "
                f"Overriding traffic light '{controlling_tl_id}' to green for this emergency!"
            )

            if controlling_tl_id is not None:
                for tid in tl_ids:
                    if tid == controlling_tl_id:
                        if current_green_tl and current_green_tl != tid:
                            if green_start_time[current_green_tl] is not None:
                                gtime = curr_time - green_start_time[current_green_tl]
                                total_green_time[current_green_tl] += gtime
                                logger.info(
                                    f"TL {current_green_tl} forced from GREEN->RED by emergency; "
                                    f"GREEN lasted {gtime:.2f}s"
                                )
                                green_start_time[current_green_tl] = None
                            current_green_tl = tid
                            last_switch_time = curr_time
                            green_start_time[tid] = curr_time

                        conn.trafficlight.setPhase(tid, 0)     #0 is green
                        conn.trafficlight.setPhaseDuration(tid, 9999)
                    else:
                        # Force all-red
                        conn.trafficlight.setPhase(tid, all_red_indices[tid])
                        conn.trafficlight.setPhaseDuration(tid, 9999)
                conn.simulationStep()
                step += 1

                r_emerg = compute_reward(tl_ids, conn)
                total_reward += r_emerg
                step_rewards.append(r_emerg)

                # If the simulation is done:
                if conn.simulation.getMinExpectedNumber() == 0:
                    logger.info("No more vehicles. Ending simulation early.")
                    break

                # Next loop iteration
                continue
        # ------------------ END EMERGENCY OVERRIDE ------------------

        time_in_current_green = (
            curr_time - last_switch_time if current_green_tl else 0.0
        )
        can_switch = time_in_current_green >= MIN_GREEN_TIME

        with torch.no_grad():
            q_values = agent(state.unsqueeze(0)).squeeze(0)

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
            logger.info(f"TL {new_tl_id} switched RED->GREEN at t={curr_time:.2f}")
            green_start_time[new_tl_id] = curr_time  # Start continuous green block

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
            if new_tl_id == current_green_tl:
                if can_switch:
                    conn.trafficlight.setPhase(new_tl_id, chosen_phase_idx)
                    conn.trafficlight.setPhaseDuration(new_tl_id, chosen_duration)
                    last_switch_time = curr_time
            else:
                if can_switch:
                    old_tl = current_green_tl
                    if green_start_time[old_tl] is not None:
                        green_duration = curr_time - green_start_time[old_tl]
                        total_green_time[old_tl] += green_duration
                        logger.info(
                            f"TL {old_tl} switched GREEN->YELLOW at t={curr_time:.2f}; "
                            f"GREEN lasted {green_duration:.2f}s"
                        )
                        green_start_time[old_tl] = None

                    conn.trafficlight.setPhase(old_tl, chosen_phase_idx)
                    conn.trafficlight.setPhaseDuration(old_tl, YELLOW_TIME)

                    for _ in range(YELLOW_TIME):
                        conn.simulationStep()
                        r_y = compute_reward(tl_ids, conn)
                        total_reward += r_y
                        step_rewards.append(r_y)
                        step += 1
                        if step >= max_steps or conn.simulation.getMinExpectedNumber() == 0:
                            break
                    if step >= max_steps or conn.simulation.getMinExpectedNumber() == 0:
                        break

                    now2 = conn.simulation.getTime()
                    for tid in tl_ids:
                        conn.trafficlight.setPhase(tid, all_red_indices[tid])
                        conn.trafficlight.setPhaseDuration(tid, ALL_RED_TIME)

                    for _ in range(ALL_RED_TIME):
                        conn.simulationStep()
                        r_r = compute_reward(tl_ids, conn)
                        total_reward += r_r
                        step_rewards.append(r_r)
                        step += 1
                        if step >= max_steps or conn.simulation.getMinExpectedNumber() == 0:
                            break
                    if step >= max_steps or conn.simulation.getMinExpectedNumber() == 0:
                        break

                    new_time = conn.simulation.getTime()
                    logger.info(f"TL {new_tl_id} switched RED->GREEN at t={new_time:.2f}")
                    green_start_time[new_tl_id] = new_time

                    conn.trafficlight.setPhase(new_tl_id, chosen_phase_idx)
                    conn.trafficlight.setPhaseDuration(new_tl_id, chosen_duration)

                    for tid in tl_ids:
                        if tid != new_tl_id:
                            conn.trafficlight.setPhase(tid, all_red_indices[tid])
                            conn.trafficlight.setPhaseDuration(tid, 9999)

                    current_green_tl = new_tl_id
                    last_switch_time = new_time

        conn.simulationStep()
        step += 1

        # Reward
        r = compute_reward(tl_ids, conn)
        total_reward += r
        step_rewards.append(r)

        # One-step TD error for logging
        next_state = get_enriched_state(tl_ids, conn)
        current_q_of_action = q_values[action_idx].item()
        with torch.no_grad():
            next_q = agent(next_state.unsqueeze(0)).squeeze(0)
        max_next_q = next_q.max().item()
        td_error = r + gamma * max_next_q - current_q_of_action
        step_losses.append(abs(td_error))

        state = next_state

        if conn.simulation.getMinExpectedNumber() == 0:
            logger.info("No more vehicles. Ending simulation early.")
            break

    # ============ Final Wrap-Up ============
    final_time = conn.simulation.getTime()
    if current_green_tl is not None and green_start_time[current_green_tl] is not None:
        green_duration = final_time - green_start_time[current_green_tl]
        total_green_time[current_green_tl] += green_duration
        logger.info(
            f"TL {current_green_tl} switched GREEN->END at t={final_time:.2f}; "
            f"GREEN lasted {green_duration:.2f}s"
        )
        green_start_time[current_green_tl] = None

    conn.close()

    logger.info(f"Final total reward over {step} steps: {total_reward:.2f}")
    for tid in tl_ids:
        logger.info(f"TL {tid} total GREEN time aggregated: {total_green_time[tid]:.2f}s")

    # Plot step-wise reward
    plt.figure(figsize=(10, 5))
    plt.plot(step_rewards, label="Reward per Step")
    plt.title("Reward vs. Time Steps (Test) with Emergency Override")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig("test_run_rewards.png")
    plt.close()

    # Plot step-wise TD error magnitude
    plt.figure(figsize=(10, 5))
    plt.plot(step_losses, label="|TD Error|")
    plt.title("TD Error Magnitude vs. Time Steps (Test) ")
    plt.xlabel("Steps")
    plt.ylabel("TD Error Magnitude")
    plt.grid(True)
    plt.legend()
    plt.savefig("test_run_td_errors.png")
    plt.close()

    # Log Aggregated Results
    logger.info(f"CO₂ Emissions (g): {cumulative_co2}")
    logger.info(f"NOx Emissions (mg): {cumulative_nox}")
    logger.info(f"PM10 Emissions (µg): {cumulative_pm10}")

    # Plot Time-Series Data
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_steps, time_series_waiting_time, label="Waiting Time (s)", linestyle='--', marker='o')
    plt.xlabel("Time Step")
    plt.ylabel("Total Waiting Time (s)")
    plt.title("Total Waiting Time Over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("waiting_time_overtime.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(time_series_steps, time_series_vehicle_count, label="Vehicle Count", linestyle='-', marker='x',
             color="orange")
    plt.xlabel("Time Step")
    plt.ylabel("Vehicle Count")
    plt.title("Vehicle Count Over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("vehicle_count_overtime.png")
    plt.close()

    # Return Results
    return {
        "time_series_waiting_time": drl_waiting_time,
        "time_series_vehicle_count": drl_vehicle_count,
        "time_series_steps": drl_steps,
        "aggregated_emissions": {
            "cumulative_co2": cumulative_co2,
            "cumulative_nox": cumulative_nox,
            "cumulative_pm10": cumulative_pm10
        }
    }


def main():
    SUMO_CONFIG_ORIGINAL = r"C:\Program Files (x86)\Eclipse\Sumo\bin\simulation13\city_network.sumocfg"

    """
    Compare DRL and Original Configurations for:
    - Time-Series Waiting Time
    - Time-Series Vehicle Count
    - Aggregated Emissions
    """
    sumoBinary = sumolib.checkBinary('sumo-gui')
    original_sumo_cmd = [
        sumoBinary,
        "-c", SUMO_CONFIG_ORIGINAL,
        "--start",
        "--quit-on-end",
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--verbose", "true",
        "--begin", str(START_TIME),
        "--end", str(END_TIME)
    ]
    traci.start(original_sumo_cmd, label="original_run")
    conn_original = traci.getConnection("original_run")
    original_waiting_time = []
    original_vehicle_count = []
    original_steps = []

    # Aggregated Metrics for Original Configuration
    orig_cumulative_co2 = 0.0
    orig_cumulative_nox = 0.0
    orig_cumulative_pm10 = 0.0

    while True:
        current_time = conn_original.simulation.getTime()
        if current_time >= END_TIME or conn_original.simulation.getMinExpectedNumber() == 0:
            break

        conn_original.simulationStep()
        veh_ids = conn_original.vehicle.getIDList()
        total_waiting_time = sum(conn_original.vehicle.getWaitingTime(vid) for vid in veh_ids)
        vehicle_count = len(veh_ids)

        original_waiting_time.append(total_waiting_time)
        original_vehicle_count.append(vehicle_count)
        original_steps.append(current_time)

        orig_cumulative_co2 += sum(conn_original.vehicle.getCO2Emission(vid) for vid in veh_ids)
        orig_cumulative_nox += sum(conn_original.vehicle.getNOxEmission(vid) for vid in veh_ids)
        orig_cumulative_pm10 += sum(conn_original.vehicle.getPMxEmission(vid) for vid in veh_ids)

    conn_original.close()

    logger.info(f"[Original] CO₂ Emissions (g): {orig_cumulative_co2}")
    logger.info(f"[Original] NOx Emissions (mg): {orig_cumulative_nox}")
    logger.info(f"[Original] PM10 Emissions (µg): {orig_cumulative_pm10}")

    # Run DRL Configuration
    drl_results = run_drl_simulation()
    drl_waiting_time = drl_results["time_series_waiting_time"]
    drl_vehicle_count = drl_results["time_series_vehicle_count"]
    drl_steps = drl_results["time_series_steps"]

    # Plot Waiting Time Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(original_steps, original_waiting_time, label="Original Config", linestyle='-', marker='x', color="blue")
    plt.plot(drl_steps, drl_waiting_time, label="DRL Config", linestyle='--', marker='o', color="green")
    plt.xlabel("Time Step")
    plt.ylabel("Total Waiting Time (s)")
    plt.title("Total Waiting Time Comparison (Original vs DRL)")
    plt.grid(True)
    plt.legend()
    plt.savefig("waiting_time_comparison.png")
    plt.close()

    # Plot Vehicle Count Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(original_steps, original_vehicle_count, label="Original Config", linestyle='-', marker='x', color="blue")
    plt.plot(drl_steps, drl_vehicle_count, label="DRL Config", linestyle='--', marker='o', color="green")
    plt.xlabel("Time Step")
    plt.ylabel("Vehicle Count")
    plt.title("Vehicle Count Comparison (Original vs DRL)")
    plt.grid(True)
    plt.legend()
    plt.savefig("vehicle_count_comparison.png")
    plt.close()

    return {
        "original": {
            "cumulative_co2": orig_cumulative_co2,
            "cumulative_nox": orig_cumulative_nox,
            "cumulative_pm10": orig_cumulative_pm10,
        },
        "drl": drl_results["aggregated_emissions"]
    }


if __name__ == "__main__":
    results = main()
    print(results)
