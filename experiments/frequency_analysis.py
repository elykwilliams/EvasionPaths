import os
import csv
import re

import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def process_csv_folder(folder_path):
    data_dict = {}

    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    strings = {}

    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            r = None

            for row in csv_reader:
                if row[0] == "Radius:":
                    r = float(row[1])
                elif row[0].startswith("(("):
                    atomic_change = row[0]
                    if atomic_change not in strings.keys():
                        strings[atomic_change] = [(r, row[1])]
                    else:
                        old_array = strings[atomic_change]
                        new_array = old_array + [(r, row[1])]
                        strings[atomic_change] = new_array

    return strings


def fill_null_values_AC(atomic_changes: dict):
    for atomic_change in atomic_changes.keys():
        count_times = atomic_changes[atomic_change]
        radii = [val[0] for val in count_times]
        null_counts = []
        for r in range(2, 42, 2):
            radius = r / 100
            if radius not in radii:
                null_counts += [(radius, 0)]
        new_count_times = count_times + null_counts
        sorted_count_times = sorted(new_count_times, key=lambda x: x[0])
        atomic_changes[atomic_change] = sorted_count_times
    return atomic_changes


def plot_values_for_string(data_dict, target_strings, time, image_path):
    plt.figure(figsize=(10, 6))
    for target_string in target_strings:
        if target_string not in data_dict:
            print(f"The string '{target_string}' does not exist in the dictionary.")
            continue

        data = data_dict[target_string]
        radii, counts = zip(*data)
        counts_str = [float(count) for count in counts]
        if target_string == '((1, 1, 2, 2), (0, 0))':
            counts_prob = [count /1000 for count in counts_str]
        else:
            counts_prob = [count * 2 /1000 for count in counts_str]

        plt.plot(radii, counts_prob, marker='o', label=f"String: {target_string}")

    plt.title(f"t={time} Count vs Radius for Multiple Strings")

    #    plt.title(f"Count vs Radius for String: {target_string}")
    plt.xlabel("Radius")
    plt.ylabel("Probability")
    plt.legend(loc='center right', title='Atomic Change', title_fontsize='14')
    plt.grid(True)
    if time < 10:
        img_file = image_path + f"time0{time}"
    else:
        img_file = image_path + f"time{time}"
    plt.savefig(img_file)
    # plt.show()

def animate_betti_curve(target_strings, data_dict):

    # Set up the figure, the axis, and the plot element
    fig, ax = plt.subplots()
    x = np.linspace(0, 0.5, 400)
    line, = ax.plot(x, np.sin(x))
    # ax.set_ylim(-0.1, 1.1)
    for target_string in target_strings:
        if target_string not in data_dict:
            print(f"The string '{target_string}' does not exist in the dictionary.")
            continue

        data = data_dict[target_string]
        radii, counts = zip(*data)
        counts = [int(count) for count in counts]

        plt.plot(radii, counts, marker='o', label=f"String: {target_string}")

    # Number of frames
    num_frames = 30
    frequencies = np.linspace(1, 10, num_frames)

    def update(frame):
        """Update function for animation"""
        line.set_ydata(np.sin(frequencies[frame] * x))
        ax.set_title(f"Sine Wave: {frequencies[frame]:.2f}Hz")
        return line,

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

    # Save the animation
    ani.save('sine_wave_animation.mp4', writer='ffmpeg', fps=5)

    # If using Jupyter Notebook, you can also display the animation inline:
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())

    plt.close()

def popular_atomic_changes(data_dict, threshold):
    keys_above_threshold = []

    for key, values in data_dict.items():
        if any(int(count) >= threshold for _, count in values):
            keys_above_threshold.append(key)

    return keys_above_threshold


def generate_graphic(folder_path, threshold_value, time, image_path):
    preprocessed_atomic_change_frequencies = process_csv_folder(folder_path)
    postprocessed_ac_freq = fill_null_values_AC(preprocessed_atomic_change_frequencies)
    all_atomic_changes = list(postprocessed_ac_freq.keys())
    most_popular_ac = popular_atomic_changes(postprocessed_ac_freq, threshold_value)
    print("Keys with counts above threshold:", most_popular_ac)
    print(len(most_popular_ac))
    plot_values_for_string(postprocessed_ac_freq, most_popular_ac, time, image_path)


if __name__ == "__main__":
    threshold = 100
    time_steps = 10000

    path = "./"

    """The following is for when you have a specific folder where you just want a single folder.
    """
    if not os.path.exists(path+"./"):
        os.makedirs(path + "./")

    generate_graphic(
        folder_path=path,
        threshold_value=threshold,
        time=time_steps,
        image_path="./")
