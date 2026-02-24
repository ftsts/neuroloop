import json
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from dbsenv.utils.synchrony import kop


def _get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception:
        return "unknown"


def rollout(agent, env):

    obs, info = env.reset()
    done = False

    actions = []
    num_samples = info["num_samples"]

    with tqdm(total=num_samples, desc="Rollout") as pbar:
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            actions.append(info["action"])

            done = terminated or truncated
            pbar.update(1)

    actions = np.array(actions).squeeze()

    episode_data = {
        "actions": actions,
        **info,
    }

    return episode_data


def evaluate(episode_data):

    step_size = episode_data["step_size"]
    duration = episode_data["duration"]
    spike_time_e = episode_data["spike_time_e"]
    ne = episode_data["num_neurons_e"]
    actions = episode_data["actions"]

    t = np.arange(0.1, duration + step_size, step_size)

    print("computing kop")
    re = kop(
        sptime=spike_time_e,
        t=t,
        step_size=step_size,
        duration=duration,
        num_neurons=ne,
    )

    window_size = 0.1  # last 10%
    idx = int(len(re) * (1 - window_size))
    kop_final = np.mean(re[idx:])

    metrics = {
        "kop_mean": np.mean(re),
        "kop_min": np.min(re),
        "kop_final": kop_final,
        "energy_l2": np.sum(actions**2),
        "energy_l1": np.sum(np.abs(actions)),
        "energy_mean": np.mean(actions**2),
    }

    episode_data["t"] = t
    episode_data["kop"] = re
    episode_data["metrics"] = metrics

    return episode_data


def save_eval(data, path=None) -> Path:

    if path is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path("./data/results").mkdir(exist_ok=True)
        path = f"./data/results/eval_{ts}.npz"

    git_hash = _get_git_hash()

    np.savez_compressed(
        path,
        actions=data["actions"],
        spike_time_e=data["spike_time_e"],
        t=data["t"],
        kop=data["kop"],
        metrics=json.dumps(data["metrics"]),
        step_size=data["step_size"],
        duration=data["duration"],
        num_neurons_e=data["num_neurons_e"],
        git_hash=git_hash,
        spike_e=data["spike_e"],
        spike_i=data["spike_i"],
        w_ie=data["w_ie"],
        j_i=data["j_i"],
    )

    print(f"Saved evaluation data to: {path}")

    return path


def load_eval(path):
    npz = np.load(path, allow_pickle=True)

    data = {
        "actions": npz["actions"],
        "spike_time_e": npz["spike_time_e"],
        "t": npz["t"],
        "kop": npz["kop"],
        "metrics": json.loads(str(npz["metrics"])),
        "step_size": float(npz["step_size"]),
        "duration": float(npz["duration"]),
        "num_neurons_e": int(npz["num_neurons_e"]),
        "git_hash": str(npz["git_hash"]),
        "spike_e": npz["spike_e"],
        "spike_i": npz["spike_i"],
        "w_ie": npz["w_ie"],
        "j_i": npz["j_i"],
    }

    return data
