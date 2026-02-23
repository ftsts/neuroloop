"""
Example usage of the FTSTS DBS environment with a PPO agent to simulate closed-loop control.
"""

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RescaleAction
from dbsenv.wrappers import DBSNormalizeObservation
from dbsenv.neural_models import EILIFNetwork
from dbsenv.utils.synchrony import kop
from dbsenv.utils import SimConfig
from neuroloop.utils import plot_kop


TB_LOG_DIR = "./ppo_tensorboard/"  # tensorboard logging directory
AGENT_PATH = "./agents/ppo_dbs_agent"  # filename to save trained agent


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


def interact(agent, env):
    raw_env = env.unwrapped
    num_samples = raw_env.model.num_samples

    obs, _ = env.reset()
    done = False
    sptime = None
    with tqdm(total=num_samples, desc="Agent Interaction") as pbar:
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            sptime = info["spike_time_e"]
            pbar.update(1)

    step_size = raw_env.model.step_size
    duration = raw_env.model.duration
    ne = raw_env.model.num_e
    t = np.arange(0.1, duration + step_size, step_size)

    print("computing kop")
    re = kop(
        sptime=sptime,
        t=t,
        step_size=step_size,
        duration=duration,
        num_neurons=ne,
    )

    print("plotting kop")
    plot_kop(t, re)


def main():
    """
    Trains and uses an SB3 PPO agent in the FTSTS Environment, simulating the
    closed-loop regime.
    """

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

    # Train the agent.
    agent.learn(total_timesteps=10_000)
    agent.save(AGENT_PATH)

    # Use the agent.
    interact(agent, env)

    env.close()


if __name__ == "__main__":
    main()
