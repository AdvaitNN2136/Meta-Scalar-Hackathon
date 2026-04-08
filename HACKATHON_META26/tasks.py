import numpy as np
from env import DisasterEnv

def run_task(task_id: str, model) -> float:
    """
    Runs a task and returns score between 0.0 and 1.0
    task_id: 'task_1_contain' | 'task_2_prevent_cascade' | 'task_3_mass_casualty'
    """
    if task_id == "task_1_contain":
        return grade_task_1(model)
    elif task_id == "task_2_prevent_cascade":
        return grade_task_2(model)
    elif task_id == "task_3_mass_casualty":
        return grade_task_3(model)
    else:
        raise ValueError(f"Unknown task: {task_id}")

# ── TASK 1: EASY — Contain single disaster ────────────────────────────
def grade_task_1(model, n_episodes=5) -> float:
    """
    Score = fraction of steps where avg severity < 0.5
    Easy: only fire disasters, no cascades
    """
    scores = []
    for _ in range(n_episodes):
        env = DisasterEnv()
        obs, _ = env.reset()

        # Force only fire disasters
        env.disaster_type[:] = 1
        env.severity = np.random.uniform(0.1, 0.3, env.n_cities).astype(np.float32)

        steps_below_threshold = 0
        total_steps = 0

        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(int(action))
            total_steps += 1

            if np.mean(info["severity"]) < 0.5:
                steps_below_threshold += 1

            if done or truncated:
                break

        scores.append(steps_below_threshold / total_steps)

    return float(np.mean(scores))


# ── TASK 2: MEDIUM — Prevent cascade events ───────────────────────────
def grade_task_2(model, n_episodes=5) -> float:
    """
    Score = 1.0 if <10 cascades, scales down to 0.0 at 50+ cascades
    Medium: mixed disasters, cascades possible
    """
    scores = []
    for _ in range(n_episodes):
        env = DisasterEnv()
        obs, _ = env.reset()

        total_cascades = 0

        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(int(action))
            total_cascades += len(info["cascade_events"])
            if done or truncated:
                break

        # Score: 1.0 = 0 cascades, 0.0 = 50+ cascades
        score = max(0.0, 1.0 - (total_cascades / 50.0))
        scores.append(score)

    return float(np.mean(scores))


# ── TASK 3: HARD — Mass casualty minimization ─────────────────────────
def grade_task_3(model, n_episodes=5) -> float:
    """
    Score based on casualties, severity, disease risk, infrastructure
    Hard: all disaster types active, storm weather, compound disasters
    """
    scores = []
    for _ in range(n_episodes):
        env = DisasterEnv()
        obs, _ = env.reset()

        # Hard mode: force compound disasters and storm
        env.disaster_type = np.array([1, 2, 3, 4, 6, 2, 5, 1])
        env.severity = np.random.uniform(0.3, 0.6, env.n_cities).astype(np.float32)
        env.weather_state = 2  # storm

        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(int(action))
            if done or truncated:
                final_info = info
                break

        # Score components
        severity_score      = 1.0 - float(np.mean(final_info["severity"]))
        disease_score       = 1.0 - float(np.mean(final_info["disease_risk"]))
        infra_score         = 1.0 - float(np.mean(final_info["infrastructure_dmg"]))
        casualty_score      = max(0.0, 1.0 - final_info["total_casualties"] / 5000.0)

        # Weighted final score
        score = (
            severity_score  * 0.35 +
            casualty_score  * 0.35 +
            disease_score   * 0.15 +
            infra_score     * 0.15
        )
        scores.append(float(np.clip(score, 0.0, 1.0)))

    return float(np.mean(scores))


if __name__ == "__main__":
    from stable_baselines3 import PPO

    print("Loading model...")
    model = PPO.load("disaster_ppo_final")

    for task in ["task_1_contain", "task_2_prevent_cascade", "task_3_mass_casualty"]:
        score = run_task(task, model)
        print(f"{task}: {score:.4f}")
