"""
Using the FTSTS DBS environment with a fixed action to simulate the original open-loop regime.
"""

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from dbsenv.wrappers import DBSNormalizeObservation
from dbsenv.neural_models import EILIFNetwork
from dbsenv.utils import SimConfig
from neuroloop.evaluation import evaluate
from neuroloop.utils import plot_kop, plot_action


def env_config():
    """Configure Environment."""

    # Configure simulation.
    sim_config = SimConfig(
        duration=5000,
        step_size=0.1,
        sample_duration=20,
    )

    # Create the environment.
    env = gym.make(
        id="dbsenv/DBS-FTSTS-v0",
        sim_config=sim_config,
        model_class=EILIFNetwork,
        model_params={
            "num_e": 160,
            "num_i": 40,
        },
    )
    env = DBSNormalizeObservation(env)  # deterministic scaling
    env = NormalizeObservation(env)  # running mean/std
    env = NormalizeReward(env)

    return env


def open_loop_rollout(action, env):

    obs, info = env.reset()
    done = False

    sptime = None
    actions = []
    num_samples = info["num_samples"]

    with tqdm(total=num_samples, desc="Rollout") as pbar:
        while not done:
            _action = np.array([action], dtype=np.float64)
            obs, reward, terminated, truncated, info = env.step(_action)

            actions.append(info["action"])
            sptime = info["spike_time_e"]

            done = terminated or truncated
            pbar.update(1)

    actions = np.array(actions)

    episode_data = {
        "actions": actions,
        "sptime": sptime,
        "step_size": info["step_size"],
        "duration": info["duration"],
        "num_neurons_e": info["num_neurons_e"],
    }

    return episode_data


def plot_results(episode_data):
    t = episode_data["t"]
    re = episode_data["kop"]
    plot_kop(t, re)

    actions = episode_data["actions"]
    plot_action(actions)


def main():
    """Simulates the original open-loop ftsts regime."""

    # Create the environment.
    env = env_config()

    # Open-Loop Simulation.
    v_stim = 100  # mV
    episode_data = open_loop_rollout(v_stim, env)

    # Evaluate.
    episode_data = evaluate(episode_data)

    # Print Metrics.
    print("\n--- Evaluation Metrics ---")
    for k, v in episode_data["metrics"].items():
        print(f"  {k}: {v:.4f}")

    plot_results(episode_data)

    env.close()


if __name__ == '__main__':
    main()
