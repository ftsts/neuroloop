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
from neuroloop.evaluation import evaluate, save_eval, load_eval
from neuroloop.plots import (
    plot_kop,
    plot_action,
    plot_synchrony,
    plot_spike_patterns,
    plot_avg_synaptic_weight,
)


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

    actions = []
    num_samples = info["num_samples"]

    with tqdm(total=num_samples, desc="Rollout") as pbar:
        while not done:
            _action = np.array([action], dtype=np.float64)
            obs, reward, terminated, truncated, info = env.step(_action)

            actions.append(info["action"])

            done = terminated or truncated
            pbar.update(1)

    actions = np.array(actions)

    episode_data = {
        "actions": actions,
        **info,
    }

    return episode_data


def plot_results(data):
    t = data["t"]
    re = data["kop"]
    plot_kop(t, re)

    actions = data["actions"]
    plot_action(actions)

    # sptime, step_size, duration, ne, J_I, W_IE, synchrony, spike_e, spike_i, si = data
    # t = np.arange(0.1, duration + step_size, step_size)
    # t = np.ascontiguousarray(t, dtype=np.float64)
    # plot_synchrony(synchrony)
    # plot_synchrony(si)

    spike_e = data["spike_e"]
    spike_i = data["spike_i"]
    step_size = data["step_size"]
    plot_spike_patterns(spike_e, spike_i, step_size)

    j_i = data["j_i"]
    w_ie = data["w_ie"]
    duration = data["duration"]
    plot_avg_synaptic_weight(t, j_i, w_ie, duration)


def main():
    """Simulates the original open-loop ftsts regime."""

    # Create the environment.
    env = env_config()
    v_stim = 100  # (mV) simulate open-loop control

    episode_data = open_loop_rollout(v_stim, env)
    episode_data = evaluate(episode_data)

    path = save_eval(episode_data)
    data = load_eval(path)

    print("\n--- Evaluation Metrics ---")
    print()
    for k, v in data["metrics"].items():
        print(f"  {k}: {v:.4f}")

    plot_results(data)

    env.close()


if __name__ == "__main__":
    main()
