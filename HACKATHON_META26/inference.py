"""
Inference Script — Disaster Response Management System
Follows OpenEnv [START] [STEP] [END] format strictly.
"""
import os
import sys
import json
import numpy as np
from stable_baselines3 import PPO
from env import DisasterEnv
from tasks import run_task

# ── Load model ────────────────────────────────────────────────────────
try:
    model = PPO.load("disaster_ppo_final")
    print("# Loaded model: disaster_ppo_final", flush=True)
except Exception as e:
    print(f"# Error loading model: {e}", flush=True)
    sys.exit(1)

TASKS = ["task_1_contain", "task_2_prevent_cascade", "task_3_mass_casualty"]

def run_task_with_logging(task_id: str, model) -> float:
    env = DisasterEnv()
    obs, _ = env.reset()

    # Task-specific setup
    if task_id == "task_1_contain":
        env.disaster_type[:] = 1
        env.severity = np.random.uniform(0.1, 0.3, env.n_cities).astype(np.float32)
    elif task_id == "task_3_mass_casualty":
        env.disaster_type = np.array([1, 2, 3, 4, 6, 2, 5, 1])
        env.severity = np.random.uniform(0.3, 0.6, env.n_cities).astype(np.float32)
        env.weather_state = 2

    # START payload
    start_payload = {
        "task_id":        task_id,
        "observation":    obs.tolist(),
        "state":          env.state(),
        "action_n":       int(env.action_space.n),
    }
    print(f"[START] {json.dumps(start_payload)}", flush=True)

    total_reward = 0.0
    step_num = 0

    for step_num in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)

        # STEP payload
        step_payload = {
            "step":        step_num,
            "action":      action,
            "reward":      round(float(reward), 4),
            "observation": obs.tolist(),
            "done":        bool(done or truncated),
            "state": {
                "severity":        [round(float(x), 3) for x in info["severity"]],
                "disaster_names":  info["disaster_names"],
                "teams_available": int(info["teams_available"]),
                "total_saved":     int(info["total_saved"]),
                "cascades":        len(info["cascade_events"]),
                "weather":         info["weather"],
                "time":            info["time_of_day"],
            }
        }
        print(f"[STEP] {json.dumps(step_payload)}", flush=True)

        if done or truncated:
            break

    # Compute final score
    score = run_task(task_id, model)

    # END payload
    end_payload = {
        "task_id":       task_id,
        "score":         round(float(score), 4),
        "total_reward":  round(float(total_reward), 4),
        "total_steps":   step_num + 1,
        "total_saved":   int(env.total_saved),
        "total_casualties": int(env.total_casualties),
    }
    print(f"[END] {json.dumps(end_payload)}", flush=True)

    return score

def main():
    for task_id in TASKS:
        run_task_with_logging(task_id, model)

if __name__ == "__main__":
    main()
