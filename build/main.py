# main.py

import threading
import uvicorn
import sumolib
import traci
import signal
import sys

from fastapi_app import app
from test_thread import test_script_thread
from inference_thread import inference_script_thread
from shared_data import shared_metrics, shared_metrics_lock

SUMO_CONFIG_PATH = r"C:\Users\Azd_A\OneDrive\Desktop\simulation_test\city_network.sumocfg"


def run_fastapi_app():
    """
    Launch Uvicorn server for the FastAPI app in this same script.
    By default, this call is blocking, so we use it in a separate thread.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def signal_handler(sig, frame):
    """
    Handle Keyboard Interrupt (Ctrl+C) by marking the simulation status as 'stopped',
    and then exiting the program.
    """
    print('You pressed Ctrl+C!')
    with shared_metrics_lock:
        shared_metrics["status"] = "stopped"

    print("Exiting...")
    sys.exit(0)


def main():
    # 1) Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # 2) Start SUMO once in main
    sumoBinary = sumolib.checkBinary('sumo-gui')  # or 'sumo' if you want headless
    sumo_cmd = [
        sumoBinary,
        "-c", SUMO_CONFIG_PATH,
        "--start",
        "--quit-on-end",
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--verbose", "true",
        "--begin", "0",
        "--end", "20000"
    ]
    traci.start(sumo_cmd, label="shared_conn")
    print("SUMO started with label='shared_conn'.")

    # 3) Create & start the threads
    t1 = threading.Thread(target=test_script_thread, name="TestScriptThread")
    t2 = threading.Thread(target=inference_script_thread, name="InferenceScriptThread")
    t3 = threading.Thread(target=run_fastapi_app, name="FastAPIThread")

    t1.start()
    t2.start()
    t3.start()

    # 4) Wait for RL threads to finish. (We might leave the FastAPI alive.)
    t1.join()
    t2.join()

    # 5) Once RL threads are done, close the SUMO connection
    traci.getConnection("shared_conn").close()
    print("Test & inference threads finished. SUMO connection closed.")
    print("FastAPI server is still running in thread t3 (unless you stop it).")



# replay_buffer.py

import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        self.tree[parent] += change
        if parent > 1:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx
        right = left + 1
        if left >= 2 * self.capacity:
            return idx
        if self.tree[left] >= s:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[1]

    def add(self, p, data):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.tree = SumTree(capacity)

    def _get_priority(self, td_error):
        return (abs(td_error) + 1e-5) ** self.alpha

    def add(self, td_error, sample):
        p = self._get_priority(td_error)
        self.tree.add(p, sample)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            s = random.random() * segment + i * segment
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        total = self.tree.total()
        sampling_probs = np.array(priorities) / total
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()

        return idxs, batch, weights

    def update(self, idx, td_error):
        p = self._get_priority(td_error)
        self.tree.update(idx, p)


if __name__ == "__main__":
    main()

