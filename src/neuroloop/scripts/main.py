from pathlib import Path
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RescaleAction
from dbsenv.wrappers import DBSNormalizeObservation
from dbsenv.neural_models import EILIFNetwork
from dbsenv.utils import SimConfig
from neuroloop.evaluation import rollout, evaluate, save_eval, load_eval
from neuroloop.plots import (
    plot_kop,
    plot_action,
    plot_synchrony,
    plot_spike_patterns,
    plot_avg_synaptic_weight,
)

TB_LOG_DIR = Path("./ppo_tensorboard/")  # tensorboard logging directory
AGENTS_DIR = Path("./agents/")  # for saving trained agents


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
        id='dbsenv/DBS-FTSTS-v0',
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
    env = RescaleAction(env, min_action=-1, max_action=1)

    return env


def agent_config(env, path: Path = None, load=False):
    if load:
        assert path.exists() and path.is_file(), path
        agent = PPO.load(path)
        return agent

    agent = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.5,
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
    )

    agent.learn(total_timesteps=10_000)

    if path is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = AGENTS_DIR / f"ppo_ftsts_{ts}.zip"

    print(f"Agent saved to: {path}")
    agent.save(path)

    return agent


def plot_results(episode_data: dict):
    t = episode_data["t"]
    re = episode_data["kop"]
    plot_kop(t, re)

    actions = episode_data["actions"]
    plot_action(actions)

    # sptime, step_size, duration, ne, J_I, W_IE, synchrony, spike_e, spike_i, si = data
    # t = np.arange(0.1, duration + step_size, step_size)
    # t = np.ascontiguousarray(t, dtype=np.float64)
    # plot_synchrony(synchrony)
    # plot_synchrony(si)

    spike_e = episode_data["spike_e"]
    spike_i = episode_data["spike_i"]
    step_size = episode_data["step_size"]
    plot_spike_patterns(spike_e, spike_i, step_size)

    j_i = episode_data["j_i"]
    w_ie = episode_data["w_ie"]
    duration = episode_data["duration"]
    plot_avg_synaptic_weight(t, j_i, w_ie, duration)


def main():
    """
    Trains and uses an SB3 PPO agent in the FTSTS Environment, simulating the
    closed-loop regime.
    """
    assert AGENTS_DIR.exists() and AGENTS_DIR.is_dir()
    assert TB_LOG_DIR.exists() and TB_LOG_DIR.is_dir()

    env = env_config()
    agent = agent_config(env)  # train or load trained agent

    episode_data = rollout(agent, env)  # agent interacts with env
    episode_data = evaluate(episode_data)

    path = save_eval(episode_data)
    results = load_eval(path)

    print("\n--- Evaluation Metrics ---")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}")

    plot_results(results)

    env.close()
    del agent


if __name__ == "__main__":
    main()
