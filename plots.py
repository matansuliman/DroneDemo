import math
import matplotlib.pyplot as plt

from logger import LOGGER
from config import CONFIG

def plot_safe(ax, x, y, *args, **kwargs):
    """plot but truncate to the shortest length"""
    min_len = min(len(x), len(y))
    ax.plot(x[:min_len], y[:min_len], *args, **kwargs)

def plot(logs_dict, ext = CONFIG["plotter"]["ext"]):

    for name, log in logs_dict.items():
        list_keys = list(log.keys())
        n_signals = len(list_keys) - 1

        n_cols = 3
        n_rows = math.ceil(n_signals / n_cols)

        fig_width = 5 * n_cols
        fig_height = 2.5 * n_rows

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axs = axs.flatten()

        x_label = list_keys[0]
        headlines = list_keys[1:]
        for i, headline in enumerate(headlines):
            plot_safe(axs[i], log[x_label], log[headline])
            axs[i].set_title(headline)

            axs[i].grid(True)
            axs[i].legend()
            axs[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))

        plt.tight_layout()
        filename = f"{name}-plot.{ext}"
        plt.savefig(filename)
        LOGGER.info(f"Plots: plotted logs to {filename}")

    LOGGER.info(f"Plots: finished")