# inference_thread.py

import os
import csv
import numpy as np
import tensorflow as tf
from keras.models import load_model
import traci

from rl_helpers import update_shared_metrics
from shared_data import shared_inference, shared_inference_lock
import time  # Possibly used for debugging or sleeps

import pandas as pd

def inference_script_thread():
    """
    Thread function for real_time_inference.py-like logic.
    Uses 'shared_conn' for SUMO and updates shared_inference.
    """
    group_names = ['A', 'B', 'C', 'D']
    loaded_models = {}

    for group_name in group_names:
        model_path = f"C:\\Users\\Azd_A\\OneDrive\\Desktop\\SUMO\\model\\model_{group_name}.h5"
        if os.path.exists(model_path):
            loaded_models[group_name] = load_model(
                model_path,
                custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
            )
        else:
            print(f"[Warning] No model file found for {group_name} at {model_path}")

    # Configuration
    TIME_WINDOW_SEC = 5
    MAX_DETS = 12
    NUM_FEATURES = 7

    group_data_buffers = {g: {} for g in group_names}
    group_detector_ids = {
        'A': [
            'e1det_412441780#1_0',
            'e1det_412441780#1_1'
        ],
        'B': [
            'e1det_688169644#0_0',
            'e1det_688169644#0_1',
            'e1det_688169644#0_2',
            'e1det_688169644#0_3'
        ],
        'C': [
            'e1det_688171462#2_0',
            'e1det_688171462#2_1',
            'e1det_688171462#2_2'
        ],
        'D': [
            'e1det_688214484#0_0',
            'e1det_688214484#0_1',
            'e1det_688214484#0_2'
        ]
    }

    conn_1 = traci.getConnection("shared_conn")

    def get_current_detector_features(conn, group_name, det_id, current_step):
        """
        Retrieve features from an E1 detector (induction loop).
        Return a dict of exactly 7 features.
        """
        flow = conn.inductionloop.getLastStepVehicleNumber(det_id)
        speed = conn.inductionloop.getLastStepMeanSpeed(det_id)
        occupancy = conn.inductionloop.getLastStepOccupancy(det_id)
        mean_halting_duration = 0.0  # placeholders
        jam_length = 0.0  # placeholders

        return {
            "speed": speed,
            "flow": flow,
            "occupancy": occupancy,
            "meanHaltingDuration": mean_halting_duration,
            "meanMaxJamLengthInMeters": jam_length,
            "meanSpeed": speed,
            "maxOccupancy": 0.0  # will fill in from sliding window
        }

    def update_sliding_time_window(group, det_id, features_dict, buffer_dict, current_step):
        """Add (current_step, features_dict) to a buffer; remove data older than TIME_WINDOW_SEC."""
        if det_id not in buffer_dict[group]:
            buffer_dict[group][det_id] = []
        buffer_dict[group][det_id].append((current_step, features_dict))

        cutoff_step = current_step - TIME_WINDOW_SEC
        buffer_dict[group][det_id] = [
            (t, f) for (t, f) in buffer_dict[group][det_id] if t > cutoff_step
        ]

    def aggregate_and_prepare_window(buffer_dict, group, current_step):
        """Collect the most recent data for all detectors in `group` => shape (1, MAX_DETS, 7)."""
        det_ids = group_detector_ids[group]
        latest_records = []

        for det_id in det_ids:
            if det_id not in buffer_dict[group]:
                continue
            if not buffer_dict[group][det_id]:
                continue

            # Last record for this detector
            _, latest_features = buffer_dict[group][det_id][-1]

            # Fill maxOccupancy from entire window for this detector
            all_occupancies = [item[1]["occupancy"] for item in buffer_dict[group][det_id]]
            max_occ = max(all_occupancies) if len(all_occupancies) > 0 else 0.0
            latest_features["maxOccupancy"] = max_occ

            numeric_features = [
                latest_features["speed"],
                latest_features["flow"],
                latest_features["occupancy"],
                latest_features["meanHaltingDuration"],
                latest_features["meanMaxJamLengthInMeters"],
                latest_features["meanSpeed"],
                latest_features["maxOccupancy"]
            ]
            latest_records.append(numeric_features)

        if not latest_records:
            return None

        # Pad or truncate to MAX_DETS
        while len(latest_records) < MAX_DETS:
            latest_records.append([0.0]*NUM_FEATURES)
        latest_records = latest_records[:MAX_DETS]

        data_2d = np.array(latest_records, dtype=np.float32)
        data_3d = data_2d[np.newaxis, ...]  # shape (1, MAX_DETS, 7)
        return data_3d

    step = 0
    try:
        while conn_1.simulation.getMinExpectedNumber() > 0:
            conn_1.simulationStep()
            step += 1

            # Update global metrics so they're not zero
            update_shared_metrics(conn_1, step)

            # 1) Update buffers
            for group_name in group_names:
                det_ids = group_detector_ids[group_name]
                for det_id in det_ids:
                    features_dict = get_current_detector_features(
                        conn_1, group_name, det_id, step
                    )
                    update_sliding_time_window(
                        group_name, det_id, features_dict,
                        group_data_buffers, step
                    )

            # 2) Inference if model is loaded
            for group_name in group_names:
                if group_name not in loaded_models:
                    continue

                data_for_model = aggregate_and_prepare_window(
                    group_data_buffers, group_name, step
                )
                if data_for_model is None:
                    continue

                if data_for_model.shape != (1, MAX_DETS, NUM_FEATURES):
                    print(f"[Warning] shape {data_for_model.shape} for group {group_name}")
                    continue

                prediction = loaded_models[group_name].predict(data_for_model)
                pred_value = float(prediction[0][0])
                print(f"[Step={step} | Group={group_name}] Prediction: {pred_value:.4f}")

                # Store the latest inference result for that group
                with shared_inference_lock:
                    shared_inference[group_name] = pred_value

                # 3) Optionally write predictions to CSV
                file_exists = os.path.exists('predictions.csv')
                with open('predictions.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(['step', 'group', 'prediction', 'time_diff'])
                        writer.writerow([step, group_name, pred_value, 0])
                    else:
                        try:
                            last_entry = pd.read_csv('predictions.csv').tail(1)
                            if not last_entry.empty:
                                last_step = int(last_entry['step'].values[0])
                                time_diff = step - last_step
                            else:
                                time_diff = 0
                        except Exception as e:
                            time_diff = 0
                            print(f"Error reading predictions.csv: {e}")
                        writer.writerow([step, group_name, pred_value, time_diff])

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
