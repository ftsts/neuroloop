import matplotlib.pyplot as plt


def plot_kop(t, re):
    """Plot the Kuramoto (Synchrony) Order Parameter."""

    plt.figure(figsize=(10, 5))
    plt.plot(t / 1000, re, label="Kuramoto Order Parameter", color="blue")
    plt.title("Time Series Kuramoto Order Parameter", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("KOP", fontsize=12)
    plt.xlim([0.0, 4.9])
    plt.ylim([0, 1.1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
