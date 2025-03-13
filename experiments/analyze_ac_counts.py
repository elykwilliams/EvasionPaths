import os
import csv
import matplotlib.pyplot as plt
import mplcursors


def load_data_from_folder(folder_path, num_sensors, total_counts):
    radii = []
    data_by_key = {}

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(f'num_{num_sensors}.csv')]
    csv_files.sort()

    for fname in csv_files:
        filepath = os.path.join(folder_path, fname)
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # rows[0] should be something like ["Radius:", "0.45"]
            # rows[1] might be ["Number:", "30"] (we don't strictly need that)
            # rows[2] should be ["Element", "Count"]

            first_line = rows[0]
            # Typically, first_line[1] should be the numeric radius
            radius_value = float(first_line[1])
            radii.append(radius_value)

            for row in rows[3:]:
                key_str = row[0]
                count = int(row[1])
                prob = count / total_counts

                if key_str not in data_by_key:
                    data_by_key[key_str] = {}

                data_by_key[key_str][radius_value] = prob

    radii = sorted(radii)

    final_data_by_key = {}
    for key_str, radius_dict in data_by_key.items():
        # For each radius in `radii`, get the count if it exists, else 0
        counts_for_key = [radius_dict.get(r, 0) for r in radii]
        final_data_by_key[key_str] = counts_for_key

    return radii, final_data_by_key


def plot_element_counts(radii, data_by_key):

    plt.figure(figsize=(10, 6))

    for key_str, counts in data_by_key.items():
        plt.plot(radii, counts, label=key_str, marker='o')  # or no marker if too many lines

    plt.xlabel("Radius")
    plt.ylabel("Count")
    plt.title("Element Observations by Radius")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # place legend outside
    plt.tight_layout()

    # Everything below is for interactive key analysis of the atomic changes.
    crs = mplcursors.cursor(hover=True)  # turn on hover

    @crs.connect("add")
    def _(sel):
        # sel.target is the (x, y) data point
        # sel.index is the index in the data
        line = sel.artist  # which matplotlib Artist (Line2D) we hit
        x, y = sel.target  # numeric data
        # If you stored your “keys” in the line’s properties,
        # you can retrieve them here:
        key_str = line.get_label()  # if you had label=key_str
        sel.annotation.set_text(f"{key_str}\n({x:.3f}, {y:.3f})")

    plt.show()


# ---- Main usage ----
if __name__ == "__main__":
    folder = "output/atomic_change_counts/"
    radii, data_by_key = load_data_from_folder(folder, num_sensors=30, total_counts=1000)
    plot_element_counts(radii, data_by_key)
