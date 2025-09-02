import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger("app")


def plot_log(drone_log, platform_log, filename="output_plot.png"):

    fig, axs = plt.subplots(4, 3, figsize=(15, 6))
    axs = axs.flatten()

    x_labels = ['x', 'y', 'z',
               'x_noise', 'y_noise', 'z_noise',
              'pitch', 'roll', 'yaw',
              'pitch_cmd', 'roll_cmd', 'yaw_cmd']
    
    y_labels = ['Centimeters', 'Centimeters', 'Centimeters',
               'Centimeters', 'Centimeters', 'Centimeters',
               'Radians', 'Radians', 'Radians',
               'Radians', 'Radians', 'Radians']

    for i, (x_label, y_label) in enumerate(zip(x_labels, y_labels)):
        scale = 100 if i < 3 else 1
        axs[i].plot(drone_log['time'], np.round(scale * np.array(drone_log[x_label]), 6), label='drone')
        if x_label in platform_log:
            axs[i].plot(platform_log['time'], np.round(scale * np.array(platform_log[x_label]), 6), '--', label='platform')
        axs[i].set_title(f"{x_label.upper()} Position" if i < 3 else x_label.capitalize())
        axs[i].set_ylabel(y_label)
        if i >= 3:
            axs[i].set_xlabel("Time (s)")
        axs[i].grid(True)
        axs[i].legend()
        axs[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))

    plt.tight_layout()
    plt.savefig(filename)