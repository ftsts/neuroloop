from pathlib import Path
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RescaleAction
from dbsenv.wrappers import DBSNormalizeObservation
from dbsenv.neural_models import EILIFNetwork
from dbsenv.utils import SimConfig
from neuroloop.evaluation import rollout, evaluate
from neuroloop.utils import plot_kop, plot_action


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


def plot_results(episode_data):
    t = episode_data["t"]
    re = episode_data["kop"]
    plot_kop(t, re)

    actions = episode_data["actions"]
    plot_action(actions)


def main():
    """
    Trains and uses an SB3 PPO agent in the FTSTS Environment, simulating the
    closed-loop regime.
    """
    assert AGENTS_DIR.exists() and AGENTS_DIR.is_dir()
    assert TB_LOG_DIR.exists() and TB_LOG_DIR.is_dir()

    # Configure the Environment.
    env = env_config()

    # Configure the Agent.
    agent = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.5,
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
    )
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_path = AGENTS_DIR / f"ppo_ftsts_{ts}"

    # Train the agent.
    agent.learn(total_timesteps=1)
    agent.save(agent_path)

    # agent = PPO.load(agent_path)

    # Rollout.
    episode_data = rollout(agent, env)

    # Evaluate.
    episode_data = evaluate(episode_data)

    # Print Metrics.
    print("\n--- Evaluation Metrics ---")
    for k, v in episode_data["metrics"].items():
        print(f"  {k}: {v:.4f}")

    plot_results(episode_data)

    env.close()
    del agent


if __name__ == "__main__":
    main()
