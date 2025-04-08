
import numpy as np
import matplotlib.pyplot as plt

def plot_log(drone_log, platform_log):
    fig, axs = plt.subplots(3, 3, figsize=(15, 6))
    axs = axs.flatten()

    labels = ['x', 'y', 'z',
              'pitch', 'roll', 'yaw',
              'pitch_cmd', 'roll_cmd', 'yaw_cmd']
    ylabels = ['Centimeters', 'Centimeters', 'Centimeters',
               'Radians', 'Radians', 'Radians',
               'Radians', 'Radians', 'Radians']

    for i, key in enumerate(labels):
        scale = 100 if i < 3 else 1
        axs[i].plot(drone_log['time'], np.round(scale * np.array(drone_log[key]), 6), label='drone')
        if key in platform_log:
            axs[i].plot(platform_log['time'], np.round(scale * np.array(platform_log[key]), 6), '--', label='platform')
        axs[i].set_title(f"{key.upper()} Position" if i < 3 else key.capitalize())
        axs[i].set_ylabel(ylabels[i])
        if i >= 3:
            axs[i].set_xlabel("Time (s)")
        axs[i].grid(True)
        axs[i].legend()
        axs[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))

    plt.tight_layout()
    filename = "output_plot.png"
    plt.savefig(filename)