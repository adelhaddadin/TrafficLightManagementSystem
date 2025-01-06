import os
import sys
import random
import logging
import copy
from typing import List, Tuple
import traci
import sumolib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from replay_memory import PrioritizedReplayBuffer
from agent import EnhancedDQNAgent

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
POSSIBLE_DURATIONS = [5,10, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
NUM_DURATIONS = len(POSSIBLE_DURATIONS)
RED_PHASE_DURATION = 10
MAX_SINGLE_GREEN_TIME = 180

class TrafficLightDRLSimulationPrioritizedNoRepick:
    def __init__(self, sumo_config_path: str):
        self.sumo_config_path = sumo_config_path
        self.num_episodes = 350
        self.max_steps = 20000
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.min_green_time = 20
        self.last_switch_time = 0
        self.current_green_light = None
        self.prev_chosen_tl_idx = None

        self.sumo_cmd = self._prepare_sumo_cmd()
        self.num_phases = None

        self.replay_capacity = 10000
        self.memory = PrioritizedReplayBuffer(self.replay_capacity, alpha=0.6)
        self.beta_start = 0.4
        self.beta_frames = self.num_episodes * self.max_steps
        self.frame_count = 0

        self.target_update_interval = 1000
        self.step_counter = 0

        self.total_rewards = []

    def _prepare_sumo_cmd(self):
        sumoBinary = sumolib.checkBinary('sumo-gui')
        return [
            sumoBinary,
            "-c", self.sumo_config_path,
            "--start",
            "--quit-on-end",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--verbose", "true",
            "--begin", "0",
            "--end", "20000"
        ]

    def _beta_by_frame(self, frame):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * frame / self.beta_frames)

    def get_enriched_state(self, tl_ids, conn) -> torch.Tensor:
        """
        Build a state vector with:
          - For each intersection:
             1) queue length per lane, normalized by global max queue
             2) vehicle count per lane, normalized by global max count
             3) 'is_stuck' flag (1 if queue > 10 else 0)
          - Uses global maximum across all intersections for normalization
        """
        # First pass: find global max queue and count
        max_queue_global = 1e-8
        max_count_global = 1e-8

        for tl_id in tl_ids:
            lanes = conn.trafficlight.getControlledLanes(tl_id)
            local_queues = [conn.lane.getLastStepHaltingNumber(l) for l in lanes]
            local_counts = [conn.lane.getLastStepVehicleNumber(l) for l in lanes]
            max_queue_global = max(max_queue_global, max(local_queues))
            max_count_global = max(max_count_global, max(local_counts))

        # Second pass: build the normalized + is_stuck vector
        state = []
        for tl_id in tl_ids:
            lanes = conn.trafficlight.getControlledLanes(tl_id)
            queue_lengths = [conn.lane.getLastStepHaltingNumber(l) for l in lanes]
            vehicle_counts = [conn.lane.getLastStepVehicleNumber(l) for l in lanes]

            normalized_queues = [q / max_queue_global for q in queue_lengths]
            normalized_counts = [c / max_count_global for c in vehicle_counts]

            is_stuck_list = [1 if q > 10 else 0 for q in queue_lengths]

            state.extend(normalized_queues)
            state.extend(normalized_counts)
            state.extend(is_stuck_list)

        return torch.tensor(state, dtype=torch.float32)

    def compute_reward(self, traffic_light_ids, traci_connection, fairness_weight=1.0) -> float:
        rewards = []
        for tl_id in traffic_light_ids:
            lanes = traci_connection.trafficlight.getControlledLanes(tl_id)
            waiting_time = sum(traci_connection.lane.getWaitingTime(lane) for lane in lanes)
            queue_length = sum(traci_connection.lane.getLastStepHaltingNumber(lane) for lane in lanes)
            intersection_penalty = -waiting_time - 2 * queue_length
            rewards.append(intersection_penalty)

        fairness_penalty = fairness_weight * np.std(rewards)
        total_reward = sum(rewards) - fairness_penalty

        return total_reward

    def get_all_red_phase_index(self, tl_id, conn):
        phases = conn.trafficlight.getAllProgramLogics(tl_id)[0].phases
        for i, p in enumerate(phases):
            if all(s == 'r' for s in p.state):
                return i
        return None

    def select_actions(self, agent, state, tl_ids, conn, current_time):
        actions = [None] * len(tl_ids)
        red_indices = {}

        # gather all-red index
        for i, tl_id in enumerate(tl_ids):
            red_idx = self.get_all_red_phase_index(tl_id, conn)
            if red_idx is None:
                raise ValueError(f"No all-red phase found for TL {tl_id}")
            red_indices[tl_id] = red_idx

        # If we currently have a green light, check min green time & optionally max green
        if self.current_green_light is not None:
            time_since_switch = current_time - self.last_switch_time
            # enforce min green
            if time_since_switch < self.min_green_time:
                idx = tl_ids.index(self.current_green_light)
                # keep that TL unchanged, others all red
                actions[idx] = None
                for j, tid in enumerate(tl_ids):
                    if j != idx:
                        actions[j] = (red_indices[tid], RED_PHASE_DURATION)
                return actions

        for i, tid in enumerate(tl_ids):
            actions[i] = (red_indices[tid], RED_PHASE_DURATION)

        q_vals = agent(state.unsqueeze(0)).squeeze(0).detach()

        # no consecutive re-pick => if we have a prev chosen TL, mask out its entire action range
        if self.prev_chosen_tl_idx is not None:
            num_actions_per_tl = self.num_phases * NUM_DURATIONS
            start = self.prev_chosen_tl_idx * num_actions_per_tl
            end = start + num_actions_per_tl
            q_vals[start:end] = -99999.0

        if random.random() < self.epsilon:
            action_idx = random.randint(0, q_vals.numel() - 1)
        else:
            action_idx = torch.argmax(q_vals).item()

        # decode action
        num_actions_per_tl = self.num_phases * NUM_DURATIONS
        tl_idx = action_idx // num_actions_per_tl
        remainder = action_idx % num_actions_per_tl
        phase = remainder // NUM_DURATIONS
        dur_i = remainder % NUM_DURATIONS
        duration = POSSIBLE_DURATIONS[dur_i]

        chosen_tl_id = tl_ids[tl_idx]
        self.current_green_light = chosen_tl_id
        self.last_switch_time = current_time
        self.prev_chosen_tl_idx = tl_idx

        for i, tid in enumerate(tl_ids):
            if i == tl_idx:
                actions[i] = (phase, duration)
            else:
                actions[i] = (red_indices[tid], RED_PHASE_DURATION)
        return actions

    def train(self):
        try:
            # 1) Initialize SUMO to gather environment size
            traci.start(self.sumo_cmd, label="init")
            conn = traci.getConnection("init")
            tl_ids = list(conn.trafficlight.getIDList())
            n_tl = len(tl_ids)
            n_phases = len(conn.trafficlight.getAllProgramLogics(tl_ids[0])[0].phases)
            self.num_phases = n_phases
            state_size = 0
            for tid in tl_ids:
                lanes = conn.trafficlight.getControlledLanes(tid)
                state_size += (len(lanes) * 3)
            conn.close()
            agent = EnhancedDQNAgent(state_size, n_tl, n_phases)
            target = copy.deepcopy(agent)
            optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate)
            # criterion = nn.SmoothL1Loss()

            best_reward = float('-inf')

            # 2) Training episodes
            for ep in range(self.num_episodes):
                traci.start(self.sumo_cmd, label=f"ep_{ep}")
                conn = traci.getConnection(f"ep_{ep}")

                total_rew = 0.0
                self.last_switch_time = 0
                self.current_green_light = None
                self.prev_chosen_tl_idx = None
                state = self.get_enriched_state(tl_ids, conn)

                for step in range(self.max_steps):
                    self.frame_count += 1
                    curr_t = conn.simulation.getTime()
                    actions = self.select_actions(agent, state, tl_ids, conn, curr_t)
                    for tl_id, act in zip(tl_ids, actions):
                        if act is not None:
                            ph, dur = act
                            conn.trafficlight.setPhase(tl_id, ph)
                            conn.trafficlight.setPhaseDuration(tl_id, dur)

                    conn.simulationStep()

                    # 3) next state & reward
                    next_st = self.get_enriched_state(tl_ids, conn)
                    rew = self.compute_reward(tl_ids, conn)
                    total_rew += rew

                    # 4) store transition
                    if self.current_green_light is not None and self.prev_chosen_tl_idx is not None:
                        c_idx = self.prev_chosen_tl_idx
                        chosen_action = actions[c_idx]
                        if chosen_action is not None:
                            ph, dur = chosen_action
                            if dur in POSSIBLE_DURATIONS:
                                a_idx = (c_idx * self.num_phases * NUM_DURATIONS +
                                         ph * NUM_DURATIONS +
                                         POSSIBLE_DURATIONS.index(dur))
                                self.memory.add(1.0, (state, a_idx, rew, next_st))

                    # update state
                    state = next_st

                    # 5) training step
                    if self.memory.tree.n_entries >= self.batch_size:
                        beta = self._beta_by_frame(self.frame_count)
                        idxs, batch, weights = self.memory.sample(self.batch_size, beta=beta)
                        if any(sample is None for sample in batch):
                            filtered_batch = [sample for sample in batch if sample is not None]
                            if len(filtered_batch) < self.batch_size:
                                continue
                            else:
                                batch = filtered_batch

                        states, act_b, rew_b, nxt_states = zip(*batch)
                        states = torch.stack(states)
                        nxt_states = torch.stack(nxt_states)
                        rew_t = torch.tensor(rew_b, dtype=torch.float32)
                        act_t = torch.tensor(act_b, dtype=torch.long)
                        w_t = torch.tensor(weights, dtype=torch.float32)

                        current_q = agent(states)
                        with torch.no_grad():
                            next_q = target(nxt_states)
                        max_next = next_q.max(dim=1)[0]

                        target_q = rew_t + self.gamma * max_next
                        chosen_q = current_q.gather(1, act_t.unsqueeze(1)).squeeze()

                        td_errs = target_q - chosen_q
                        loss = (td_errs.abs() * w_t).mean()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        for i, idx_val in enumerate(idxs):
                            if i < len(td_errs):
                                self.memory.update(idx_val, td_errs[i].item())

                    # 6) target network update
                    self.step_counter += 1
                    if self.step_counter % self.target_update_interval == 0:
                        target.load_state_dict(agent.state_dict())

                    # break if no vehicles left (debugging)
                    if conn.simulation.getMinExpectedNumber() == 0:
                        break

                # 7) Epsilon decay
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                # 8) Check best performance
                if total_rew > best_reward:
                    best_reward = total_rew
                    torch.save({
                        'model_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epsilon': self.epsilon,
                        'memory': self.memory
                    }, "asdasdprioritized_no_repicking_best_new.pth")

                    normalized_best = best_reward / 1e6
                    logger.info(f"[PR] New best reward: {best_reward:.2f} "
                                f"(normalized: {normalized_best:.2f}), "
                                f"Episode={ep + 1}, Epsilon={self.epsilon:.4f}")

                normalized_reward = total_rew / 1e6
                logger.info(f"[PR] Episode {ep + 1}/{self.num_episodes}, "
                            f"Reward={total_rew:.2f} (Norm={normalized_reward:.2f}), "
                            f"Epsilon={self.epsilon:.4f}")

                self.total_rewards.append(total_rew)
                conn.close()

            # finalize
            torch.save(agent.state_dict(), "prioritized_no_repicking_final.pth")
            plt.plot(self.total_rewards)
            plt.title("Prioritized Replay + No Repick (with new reward/state)")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.savefig("prioritized_no_repicking.png")
            plt.close()

        except Exception as e:
            logger.error(f"Error during prioritized run: {e}")
            raise