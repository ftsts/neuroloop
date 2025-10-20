import math
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def kop_py(sptime, t, step_size, duration, num_neurons):
    """Python Implementation."""

    total_steps = len(t)
    num_steps = int(duration / step_size)

    # Compute phases.
    phi = np.zeros((total_steps, num_neurons))
    for n in trange(
        num_neurons,
        desc="Computing Neuronal Synchrony",
        unit="neuron"
    ):
        second_spike = 0
        for i in range(num_steps - 1):
            phi[i, n] = 2 * np.pi * (t[i] - sptime[i, n])
            if sptime[i + 1, n] != sptime[i, n]:
                if second_spike == 1:
                    delt = sptime[i + 1, n] - sptime[i, n]
                    a = int(math.floor(sptime[i, n] / step_size))
                    b = int(math.floor(sptime[i + 1, n] / step_size))
                    # Ensure indices are within bounds
                    if b >= total_steps:
                        b = total_steps - 1
                    unnorm = phi[a:b + 1, n]
                    if delt != 0:
                        phi[a:b + 1, n] = unnorm / delt
                second_spike = 1

    return np.abs(np.mean(np.exp(1j * phi), axis=1))


def plot_kop(t, re):
    plt.figure(figsize=(10, 5))
    plt.plot(t / 1000, re, label='Kuramoto Order Parameter', color='blue')
    plt.title('Time Series Kuramoto Order Parameter', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('KOP', fontsize=12)
    # plt.xlim([0.2, 0.8])
    plt.ylim([0, 1.1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
