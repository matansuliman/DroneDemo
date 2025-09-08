import numpy as np
import matplotlib.pyplot as plt

from logger import LOGGER
from config import CONFIG

def plot_safe(ax, x, y, *args, **kwargs):
    """plot but truncate to the shortest length"""
    min_len = min(len(x), len(y))
    ax.plot(x[:min_len], y[:min_len], *args, **kwargs)

def plot_log(drone_log, platform_log, filename= CONFIG["plotter"]["path"]):
    fig, axs = plt.subplots(4, 3, figsize=(15, 6))
    axs = axs.flatten()

    x_label = 'Time (sec)'

    headlines = ['x_true', 'y_true', 'z_true',
                'x', 'y', 'z',
                'pitch', 'roll', 'yaw',
                'pitch_cmd', 'roll_cmd', 'yaw_cmd']
    
    y_labels = ['Centimeters', 'Centimeters', 'Centimeters',
                'Centimeters', 'Centimeters', 'Centimeters',
                'Radians', 'Radians', 'Radians',
                'Radians', 'Radians', 'Radians']

    for i, (headline, y_label) in enumerate(zip(headlines, y_labels)):
        scale = 100 if y_label == 'Centimeters' else 1
        if headline in drone_log:
            plot_safe(axs[i], drone_log[x_label], scale * np.array(drone_log[headline]), label='drone')
        if headline in platform_log:
            plot_safe(axs[i], platform_log[x_label], scale * np.array(platform_log[headline]), linestyle= 'dashed', label='platform')

        # set text
        axs[i].set_title(headline)
        axs[i].set_xlabel(x_label)
        axs[i].set_ylabel(y_label)

        axs[i].grid(True)
        axs[i].legend()
        axs[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))

    plt.tight_layout()
    plt.savefig(filename)

    LOGGER.info(f"Plots: plotted logs to {filename}")