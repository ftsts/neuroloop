import numpy as np
from tqdm import tqdm
from dbsenv.utils.synchrony import kop


def rollout(agent, env):

    obs, info = env.reset()
    done = False

    sptime = None
    actions = []
    num_samples = info["num_samples"]

    with tqdm(total=num_samples, desc="Rollout") as pbar:
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            actions.append(info["action"])
            sptime = info["spike_time_e"]

            done = terminated or truncated
            pbar.update(1)

    actions = np.array(actions).squeeze()

    episode_data = {
        "actions": actions,
        "sptime": sptime,
        "step_size": info["step_size"],
        "duration": info["duration"],
        "num_neurons_e": info["num_neurons_e"],
    }

    return episode_data


def evaluate(episode_data):

    step_size = episode_data["step_size"]
    duration = episode_data["duration"]
    sptime = episode_data["sptime"]
    ne = episode_data["num_neurons_e"]
    actions = episode_data["actions"]

    t = np.arange(0.1, duration + step_size, step_size)

    print("computing kop")
    re = kop(
        sptime=sptime,
        t=t,
        step_size=step_size,
        duration=duration,
        num_neurons=ne,
    )

    metrics = {
        "kop_mean": np.mean(re),
        "kop_min": np.min(re),
        "energy_l2": np.sum(actions**2),
        "energy_l1": np.sum(np.abs(actions)),
    }

    episode_data["t"] = t
    episode_data["kop"] = re
    episode_data["metrics"] = metrics

    return episode_data
