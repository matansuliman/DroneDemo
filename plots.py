
import numpy as np
import matplotlib.pyplot as plt

def plot_log(drone_log, platform_log):
    fig, axs = plt.subplots(4, 3, figsize=(15, 6))
    axs = axs.flatten()

    xlabels = ['x', 'y', 'z',
               'x_noise', 'y_noise', 'z_noise',
              'pitch', 'roll', 'yaw',
              'pitch_cmd', 'roll_cmd', 'yaw_cmd']
    
    ylabels = ['Centimeters', 'Centimeters', 'Centimeters',
               'Centimeters', 'Centimeters', 'Centimeters',
               'Radians', 'Radians', 'Radians',
               'Radians', 'Radians', 'Radians']

    for i, (xlabel, ylabel) in enumerate(zip(xlabels, ylabels)):
        scale = 100 if i < 3 else 1
        axs[i].plot(drone_log['time'], np.round(scale * np.array(drone_log[xlabel]), 6), label='drone')
        if xlabel in platform_log:
            axs[i].plot(platform_log['time'], np.round(scale * np.array(platform_log[xlabel]), 6), '--', label='platform')
        axs[i].set_title(f"{xlabel.upper()} Position" if i < 3 else xlabel.capitalize())
        axs[i].set_ylabel(ylabel)
        if i >= 3:
            axs[i].set_xlabel("Time (s)")
        axs[i].grid(True)
        axs[i].legend()
        axs[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.2f}'))

    plt.tight_layout()
    filename = "output_plot.png"
    plt.savefig(filename)


